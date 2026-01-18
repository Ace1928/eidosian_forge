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
def _create_cache_key(self):
    """Create a predictable cache key for the current configuration.

        The cache key is intended to be compatible with file names.
        """
    args = {'roleName': self._role_name, 'accountId': self._account_id}
    if self._sso_session_name:
        args['sessionName'] = self._sso_session_name
    else:
        args['startUrl'] = self._start_url
    args = json.dumps(args, sort_keys=True, separators=(',', ':'))
    argument_hash = sha1(args.encode('utf-8')).hexdigest()
    return self._make_file_safe(argument_hash)