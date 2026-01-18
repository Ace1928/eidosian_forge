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