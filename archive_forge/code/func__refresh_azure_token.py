import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
def _refresh_azure_token(self, config):
    if 'adal' not in globals():
        raise ImportError('refresh token error, adal library not imported')
    tenant = config['tenant-id']
    authority = 'https://login.microsoftonline.com/{}'.format(tenant)
    context = adal.AuthenticationContext(authority, validate_authority=True)
    refresh_token = config['refresh-token']
    client_id = config['client-id']
    token_response = context.acquire_token_with_refresh_token(refresh_token, client_id, '00000002-0000-0000-c000-000000000000')
    provider = self._user['auth-provider']['config']
    provider.value['access-token'] = token_response['accessToken']
    provider.value['expires-on'] = token_response['expiresOn']
    if self._config_persister:
        self._config_persister(self._config.value)