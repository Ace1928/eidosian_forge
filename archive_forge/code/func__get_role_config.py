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
def _get_role_config(self, profile_name):
    """Retrieves and validates the role configuration for the profile."""
    profiles = self._loaded_config.get('profiles', {})
    profile = profiles[profile_name]
    source_profile = profile.get('source_profile')
    role_arn = profile['role_arn']
    credential_source = profile.get('credential_source')
    mfa_serial = profile.get('mfa_serial')
    external_id = profile.get('external_id')
    role_session_name = profile.get('role_session_name')
    duration_seconds = profile.get('duration_seconds')
    role_config = {'role_arn': role_arn, 'external_id': external_id, 'mfa_serial': mfa_serial, 'role_session_name': role_session_name, 'source_profile': source_profile, 'credential_source': credential_source}
    if duration_seconds is not None:
        try:
            role_config['duration_seconds'] = int(duration_seconds)
        except ValueError:
            pass
    if credential_source is not None and source_profile is not None:
        raise InvalidConfigError(error_msg='The profile "%s" contains both source_profile and credential_source.' % profile_name)
    elif credential_source is None and source_profile is None:
        raise PartialCredentialsError(provider=self.METHOD, cred_var='source_profile or credential_source')
    elif credential_source is not None:
        self._validate_credential_source(profile_name, credential_source)
    else:
        self._validate_source_profile(profile_name, source_profile)
    return role_config