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
def _resolve_source_credentials(self, role_config, profile_name):
    credential_source = role_config.get('credential_source')
    if credential_source is not None:
        return self._resolve_credentials_from_source(credential_source, profile_name)
    source_profile = role_config['source_profile']
    self._visited_profiles.append(source_profile)
    return self._resolve_credentials_from_profile(source_profile)