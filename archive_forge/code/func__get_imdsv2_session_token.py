import abc
from dataclasses import dataclass
import hashlib
import hmac
import http.client as http_client
import json
import os
import posixpath
import re
from typing import Optional
import urllib
from urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
def _get_imdsv2_session_token(self, request):
    if request is not None and self._imdsv2_session_token_url is not None:
        headers = {'X-aws-ec2-metadata-token-ttl-seconds': _IMDSV2_SESSION_TOKEN_TTL_SECONDS}
        imdsv2_session_token_response = request(url=self._imdsv2_session_token_url, method='PUT', headers=headers)
        if imdsv2_session_token_response.status != http_client.OK:
            raise exceptions.RefreshError('Unable to retrieve AWS Session Token: {}'.format(imdsv2_session_token_response.data))
        return imdsv2_session_token_response.data
    else:
        return None