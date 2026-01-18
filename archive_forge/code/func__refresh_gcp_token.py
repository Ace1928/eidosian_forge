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
def _refresh_gcp_token(self):
    if 'config' not in self._user['auth-provider']:
        self._user['auth-provider'].value['config'] = {}
    provider = self._user['auth-provider']['config']
    credentials = self._get_google_credentials()
    provider.value['access-token'] = credentials.token
    provider.value['expiry'] = format_rfc3339(credentials.expiry)
    if self._config_persister:
        self._config_persister(self._config.value)