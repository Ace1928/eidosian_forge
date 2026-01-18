import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
class EnvProvider(CredentialProvider):
    METHOD = 'env'
    CANONICAL_NAME = 'Environment'
    ACCESS_KEY = 'AWS_ACCESS_KEY_ID'
    SECRET_KEY = 'AWS_SECRET_ACCESS_KEY'
    TOKENS = ['AWS_SECURITY_TOKEN', 'AWS_SESSION_TOKEN']
    EXPIRY_TIME = 'AWS_CREDENTIAL_EXPIRATION'

    def __init__(self, environ=None, mapping=None):
        """

        :param environ: The environment variables (defaults to
            ``os.environ`` if no value is provided).
        :param mapping: An optional mapping of variable names to
            environment variable names.  Use this if you want to
            change the mapping of access_key->AWS_ACCESS_KEY_ID, etc.
            The dict can have up to 3 keys: ``access_key``, ``secret_key``,
            ``session_token``.
        """
        if environ is None:
            environ = os.environ
        self.environ = environ
        self._mapping = self._build_mapping(mapping)

    def _build_mapping(self, mapping):
        var_mapping = {}
        if mapping is None:
            var_mapping['access_key'] = self.ACCESS_KEY
            var_mapping['secret_key'] = self.SECRET_KEY
            var_mapping['token'] = self.TOKENS
            var_mapping['expiry_time'] = self.EXPIRY_TIME
        else:
            var_mapping['access_key'] = mapping.get('access_key', self.ACCESS_KEY)
            var_mapping['secret_key'] = mapping.get('secret_key', self.SECRET_KEY)
            var_mapping['token'] = mapping.get('token', self.TOKENS)
            if not isinstance(var_mapping['token'], list):
                var_mapping['token'] = [var_mapping['token']]
            var_mapping['expiry_time'] = mapping.get('expiry_time', self.EXPIRY_TIME)
        return var_mapping

    def load(self):
        """
        Search for credentials in explicit environment variables.
        """
        access_key = self.environ.get(self._mapping['access_key'], '')
        if access_key:
            logger.info('Found credentials in environment variables.')
            fetcher = self._create_credentials_fetcher()
            credentials = fetcher(require_expiry=False)
            expiry_time = credentials['expiry_time']
            if expiry_time is not None:
                expiry_time = parse(expiry_time)
                return RefreshableCredentials(credentials['access_key'], credentials['secret_key'], credentials['token'], expiry_time, refresh_using=fetcher, method=self.METHOD)
            return Credentials(credentials['access_key'], credentials['secret_key'], credentials['token'], method=self.METHOD)
        else:
            return None

    def _create_credentials_fetcher(self):
        mapping = self._mapping
        method = self.METHOD
        environ = self.environ

        def fetch_credentials(require_expiry=True):
            credentials = {}
            access_key = environ.get(mapping['access_key'], '')
            if not access_key:
                raise PartialCredentialsError(provider=method, cred_var=mapping['access_key'])
            credentials['access_key'] = access_key
            secret_key = environ.get(mapping['secret_key'], '')
            if not secret_key:
                raise PartialCredentialsError(provider=method, cred_var=mapping['secret_key'])
            credentials['secret_key'] = secret_key
            credentials['token'] = None
            for token_env_var in mapping['token']:
                token = environ.get(token_env_var, '')
                if token:
                    credentials['token'] = token
                    break
            credentials['expiry_time'] = None
            expiry_time = environ.get(mapping['expiry_time'], '')
            if expiry_time:
                credentials['expiry_time'] = expiry_time
            if require_expiry and (not expiry_time):
                raise PartialCredentialsError(provider=method, cred_var=mapping['expiry_time'])
            return credentials
        return fetch_credentials