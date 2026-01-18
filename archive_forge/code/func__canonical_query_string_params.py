import base64
import calendar
import datetime
import functools
import hmac
import json
import logging
import time
from collections.abc import Mapping
from email.utils import formatdate
from hashlib import sha1, sha256
from operator import itemgetter
from botocore.compat import (
from botocore.exceptions import NoAuthTokenError, NoCredentialsError
from botocore.utils import (
from botocore.compat import MD5_AVAILABLE  # noqa
def _canonical_query_string_params(self, params):
    key_val_pairs = []
    if isinstance(params, Mapping):
        params = params.items()
    for key, value in params:
        key_val_pairs.append((quote(key, safe='-_.~'), quote(str(value), safe='-_.~')))
    sorted_key_vals = []
    for key, value in sorted(key_val_pairs):
        sorted_key_vals.append(f'{key}={value}')
    canonical_query_string = '&'.join(sorted_key_vals)
    return canonical_query_string