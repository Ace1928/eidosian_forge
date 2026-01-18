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
def _load_auth_provider_token(self):
    if 'auth-provider' not in self._user:
        return
    provider = self._user['auth-provider']
    if 'name' not in provider:
        return
    if provider['name'] == 'gcp':
        return self._load_gcp_token(provider)
    if provider['name'] == 'azure':
        return self._load_azure_token(provider)
    if provider['name'] == 'oidc':
        return self._load_oid_token(provider)