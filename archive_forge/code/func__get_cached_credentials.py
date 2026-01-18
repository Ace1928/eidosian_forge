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