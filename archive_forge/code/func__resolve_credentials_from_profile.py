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
def _resolve_credentials_from_profile(self, profile_name):
    profiles = self._loaded_config.get('profiles', {})
    profile = profiles[profile_name]
    if self._has_static_credentials(profile) and (not self._profile_provider_builder):
        return self._resolve_static_credentials_from_profile(profile)
    elif self._has_static_credentials(profile) or not self._has_assume_role_config_vars(profile):
        profile_providers = self._profile_provider_builder.providers(profile_name=profile_name, disable_env_vars=True)
        profile_chain = CredentialResolver(profile_providers)
        credentials = profile_chain.load_credentials()
        if credentials is None:
            error_message = 'The source profile "%s" must have credentials.'
            raise InvalidConfigError(error_msg=error_message % profile_name)
        return credentials
    return self._load_creds_via_assume_role(profile_name)