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
def _build_credentials(self, auth_response) -> google.oauth2.credentials.Credentials:
    expiration_datetime = datetime.datetime.utcfromtimestamp(int(time.time()) + auth_response['expires_in'])
    return google.oauth2.credentials.Credentials(auth_response['access_token'], refresh_token=auth_response['refresh_token'], id_token=auth_response['id_token'], token_uri=self._client_config['token_uri'], client_id=self._client_config['client_id'], client_secret=self._client_config['client_secret'], scopes=self._scopes, expiry=expiration_datetime)