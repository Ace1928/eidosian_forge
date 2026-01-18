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
def _load_from_exec_plugin(self):
    if 'exec' not in self._user:
        return
    try:
        status = ExecProvider(self._user['exec']).run()
        if 'token' not in status:
            logging.error('exec: missing token field in plugin output')
            return None
        self.token = 'Bearer %s' % status['token']
        return True
    except Exception as e:
        logging.error(str(e))