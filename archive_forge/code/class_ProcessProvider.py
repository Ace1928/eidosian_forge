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
class ProcessProvider(CredentialProvider):
    METHOD = 'custom-process'

    def __init__(self, profile_name, load_config, popen=subprocess.Popen):
        self._profile_name = profile_name
        self._load_config = load_config
        self._loaded_config = None
        self._popen = popen

    def load(self):
        credential_process = self._credential_process
        if credential_process is None:
            return
        creds_dict = self._retrieve_credentials_using(credential_process)
        if creds_dict.get('expiry_time') is not None:
            return RefreshableCredentials.create_from_metadata(creds_dict, lambda: self._retrieve_credentials_using(credential_process), self.METHOD)
        return Credentials(access_key=creds_dict['access_key'], secret_key=creds_dict['secret_key'], token=creds_dict.get('token'), method=self.METHOD)

    def _retrieve_credentials_using(self, credential_process):
        process_list = compat_shell_split(credential_process)
        p = self._popen(process_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise CredentialRetrievalError(provider=self.METHOD, error_msg=stderr.decode('utf-8'))
        parsed = botocore.compat.json.loads(stdout.decode('utf-8'))
        version = parsed.get('Version', '<Version key not provided>')
        if version != 1:
            raise CredentialRetrievalError(provider=self.METHOD, error_msg=f"Unsupported version '{version}' for credential process provider, supported versions: 1")
        try:
            return {'access_key': parsed['AccessKeyId'], 'secret_key': parsed['SecretAccessKey'], 'token': parsed.get('SessionToken'), 'expiry_time': parsed.get('Expiration')}
        except KeyError as e:
            raise CredentialRetrievalError(provider=self.METHOD, error_msg=f'Missing required key in response: {e}')

    @property
    def _credential_process(self):
        if self._loaded_config is None:
            self._loaded_config = self._load_config()
        profile_config = self._loaded_config.get('profiles', {}).get(self._profile_name, {})
        return profile_config.get('credential_process')