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
def _load_azure_token(self, provider):
    if 'config' not in provider:
        return
    if 'access-token' not in provider['config']:
        return
    if 'expires-on' in provider['config']:
        if int(provider['config']['expires-on']) < time.gmtime():
            self._refresh_azure_token(provider['config'])
    self.token = 'Bearer %s' % provider['config']['access-token']
    return self.token