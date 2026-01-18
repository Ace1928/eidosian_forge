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
class BotoProvider(CredentialProvider):
    METHOD = 'boto-config'
    CANONICAL_NAME = 'Boto2Config'
    BOTO_CONFIG_ENV = 'BOTO_CONFIG'
    DEFAULT_CONFIG_FILENAMES = ['/etc/boto.cfg', '~/.boto']
    ACCESS_KEY = 'aws_access_key_id'
    SECRET_KEY = 'aws_secret_access_key'

    def __init__(self, environ=None, ini_parser=None):
        if environ is None:
            environ = os.environ
        if ini_parser is None:
            ini_parser = botocore.configloader.raw_config_parse
        self._environ = environ
        self._ini_parser = ini_parser

    def load(self):
        """
        Look for credentials in boto config file.
        """
        if self.BOTO_CONFIG_ENV in self._environ:
            potential_locations = [self._environ[self.BOTO_CONFIG_ENV]]
        else:
            potential_locations = self.DEFAULT_CONFIG_FILENAMES
        for filename in potential_locations:
            try:
                config = self._ini_parser(filename)
            except ConfigNotFound:
                continue
            if 'Credentials' in config:
                credentials = config['Credentials']
                if self.ACCESS_KEY in credentials:
                    logger.info('Found credentials in boto config file: %s', filename)
                    access_key, secret_key = self._extract_creds_from_mapping(credentials, self.ACCESS_KEY, self.SECRET_KEY)
                    return Credentials(access_key, secret_key, method=self.METHOD)