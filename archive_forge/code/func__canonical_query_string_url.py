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
def _canonical_query_string_url(self, parts):
    canonical_query_string = ''
    if parts.query:
        key_val_pairs = []
        for pair in parts.query.split('&'):
            key, _, value = pair.partition('=')
            key_val_pairs.append((key, value))
        sorted_key_vals = []
        for key, value in sorted(key_val_pairs):
            sorted_key_vals.append(f'{key}={value}')
        canonical_query_string = '&'.join(sorted_key_vals)
    return canonical_query_string