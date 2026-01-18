import datetime
import errno
import json
import os
import requests
import sys
import time
import webbrowser
import google_auth_oauthlib.flow as auth_flows
import grpc
import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
from tensorboard.uploader import util
from tensorboard.util import tb_logging
class _LimitedInputDeviceAuthFlow:
    """OAuth flow to authenticate using the limited-input device flow.

    See:
    http://developers.google.com/identity/protocols/oauth2/limited-input-device
    """

    def __init__(self, client_config, scopes):
        self._client_config = client_config['installed']
        self._scopes = scopes

    def run(self) -> google.oauth2.credentials.Credentials:
        device_response = self._send_device_auth_request()
        prompt_message = 'To sign in with the TensorBoard uploader:\n\n1. On your computer or phone, visit:\n\n   {url}\n\n2. Sign in with your Google account, then enter:\n\n   {code}\n'.format(url=device_response['verification_url'], code=device_response['user_code'])
        print(prompt_message)
        auth_response = self._poll_for_auth_token(device_code=device_response['device_code'], polling_interval=device_response['interval'], expiration_seconds=device_response['expires_in'])
        return self._build_credentials(auth_response)

    def _send_device_auth_request(self):
        params = {'client_id': self._client_config['client_id'], 'scope': ' '.join(self._scopes)}
        r = requests.post(_DEVICE_AUTH_CODE_URI, data=params).json()
        if 'device_code' not in r:
            raise RuntimeError("There was an error while contacting Google's authorization server. Please try again later.")
        return r

    def _poll_for_auth_token(self, device_code: str, polling_interval: int, expiration_seconds: int):
        token_uri = self._client_config['token_uri']
        params = {'client_id': self._client_config['client_id'], 'client_secret': self._client_config['client_secret'], 'device_code': device_code, 'grant_type': _LIMITED_INPUT_DEVICE_AUTH_GRANT_TYPE}
        expiration_time = time.time() + expiration_seconds
        while time.time() < expiration_time:
            resp = requests.post(token_uri, data=params)
            r = resp.json()
            if 'access_token' in r:
                return r
            elif 'error' in r and r['error'] == 'authorization_pending':
                time.sleep(polling_interval)
            elif 'error' in r and r['error'] == 'slow_down':
                polling_interval = int(polling_interval * 1.5)
                time.sleep(polling_interval)
            elif 'error' in r and r['error'] == 'access_denied':
                raise PermissionError('Access was denied by user.')
            elif resp.status_code in {400, 401}:
                raise ValueError('There must be an error in the request.')
            else:
                raise RuntimeError('An unexpected error occurred while waiting for authorization.')
        raise TimeoutError('Timed out waiting for authorization.')

    def _build_credentials(self, auth_response) -> google.oauth2.credentials.Credentials:
        expiration_datetime = datetime.datetime.utcfromtimestamp(int(time.time()) + auth_response['expires_in'])
        return google.oauth2.credentials.Credentials(auth_response['access_token'], refresh_token=auth_response['refresh_token'], id_token=auth_response['id_token'], token_uri=self._client_config['token_uri'], client_id=self._client_config['client_id'], client_secret=self._client_config['client_secret'], scopes=self._scopes, expiry=expiration_datetime)