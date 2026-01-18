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
class CachedCredentialFetcher:
    DEFAULT_EXPIRY_WINDOW_SECONDS = 60 * 15

    def __init__(self, cache=None, expiry_window_seconds=None):
        if cache is None:
            cache = {}
        self._cache = cache
        self._cache_key = self._create_cache_key()
        if expiry_window_seconds is None:
            expiry_window_seconds = self.DEFAULT_EXPIRY_WINDOW_SECONDS
        self._expiry_window_seconds = expiry_window_seconds

    def _create_cache_key(self):
        raise NotImplementedError('_create_cache_key()')

    def _make_file_safe(self, filename):
        filename = filename.replace(':', '_').replace(os.sep, '_')
        return filename.replace('/', '_')

    def _get_credentials(self):
        raise NotImplementedError('_get_credentials()')

    def fetch_credentials(self):
        return self._get_cached_credentials()

    def _get_cached_credentials(self):
        """Get up-to-date credentials.

        This will check the cache for up-to-date credentials, calling assume
        role if none are available.
        """
        response = self._load_from_cache()
        if response is None:
            response = self._get_credentials()
            self._write_to_cache(response)
        else:
            logger.debug('Credentials for role retrieved from cache.')
        creds = response['Credentials']
        expiration = _serialize_if_needed(creds['Expiration'], iso=True)
        return {'access_key': creds['AccessKeyId'], 'secret_key': creds['SecretAccessKey'], 'token': creds['SessionToken'], 'expiry_time': expiration}

    def _load_from_cache(self):
        if self._cache_key in self._cache:
            creds = deepcopy(self._cache[self._cache_key])
            if not self._is_expired(creds):
                return creds
            else:
                logger.debug('Credentials were found in cache, but they are expired.')
        return None

    def _write_to_cache(self, response):
        self._cache[self._cache_key] = deepcopy(response)

    def _is_expired(self, credentials):
        """Check if credentials are expired."""
        end_time = _parse_if_needed(credentials['Credentials']['Expiration'])
        seconds = total_seconds(end_time - _local_now())
        return seconds < self._expiry_window_seconds