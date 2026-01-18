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
def _assume_role_with_web_identity(self):
    token_path = self._get_config('web_identity_token_file')
    if not token_path:
        return None
    token_loader = self._token_loader_cls(token_path)
    role_arn = self._get_config('role_arn')
    if not role_arn:
        error_msg = 'The provided profile or the current environment is configured to assume role with web identity but has no role ARN configured. Ensure that the profile has the role_arnconfiguration set or the AWS_ROLE_ARN env var is set.'
        raise InvalidConfigError(error_msg=error_msg)
    extra_args = {}
    role_session_name = self._get_config('role_session_name')
    if role_session_name is not None:
        extra_args['RoleSessionName'] = role_session_name
    fetcher = AssumeRoleWithWebIdentityCredentialFetcher(client_creator=self._client_creator, web_identity_token_loader=token_loader, role_arn=role_arn, extra_args=extra_args, cache=self.cache)
    return DeferredRefreshableCredentials(method=self.METHOD, refresh_using=fetcher.fetch_credentials)