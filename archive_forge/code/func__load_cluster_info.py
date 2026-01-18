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
def _load_cluster_info(self):
    if 'server' in self._cluster:
        self.host = self._cluster['server'].rstrip('/')
        if self.host.startswith('https'):
            base_path = self._get_base_path(self._cluster.path)
            self.ssl_ca_cert = FileOrData(self._cluster, 'certificate-authority', file_base_path=base_path).as_file()
            self.cert_file = FileOrData(self._user, 'client-certificate', file_base_path=base_path).as_file()
            self.key_file = FileOrData(self._user, 'client-key', file_base_path=base_path).as_file()
    if 'insecure-skip-tls-verify' in self._cluster:
        self.verify_ssl = not self._cluster['insecure-skip-tls-verify']