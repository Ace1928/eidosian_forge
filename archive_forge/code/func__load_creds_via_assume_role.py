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
def _load_creds_via_assume_role(self, profile_name):
    role_config = self._get_role_config(profile_name)
    source_credentials = self._resolve_source_credentials(role_config, profile_name)
    extra_args = {}
    role_session_name = role_config.get('role_session_name')
    if role_session_name is not None:
        extra_args['RoleSessionName'] = role_session_name
    external_id = role_config.get('external_id')
    if external_id is not None:
        extra_args['ExternalId'] = external_id
    mfa_serial = role_config.get('mfa_serial')
    if mfa_serial is not None:
        extra_args['SerialNumber'] = mfa_serial
    duration_seconds = role_config.get('duration_seconds')
    if duration_seconds is not None:
        extra_args['DurationSeconds'] = duration_seconds
    fetcher = AssumeRoleCredentialFetcher(client_creator=self._client_creator, source_credentials=source_credentials, role_arn=role_config['role_arn'], extra_args=extra_args, mfa_prompter=self._prompter, cache=self.cache)
    refresher = fetcher.fetch_credentials
    if mfa_serial is not None:
        refresher = create_mfa_serial_refresher(refresher)
    return DeferredRefreshableCredentials(method=self.METHOD, refresh_using=refresher, time_fetcher=_local_now)