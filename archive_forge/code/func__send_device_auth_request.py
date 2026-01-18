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
def _send_device_auth_request(self):
    params = {'client_id': self._client_config['client_id'], 'scope': ' '.join(self._scopes)}
    r = requests.post(_DEVICE_AUTH_CODE_URI, data=params).json()
    if 'device_code' not in r:
        raise RuntimeError("There was an error while contacting Google's authorization server. Please try again later.")
    return r