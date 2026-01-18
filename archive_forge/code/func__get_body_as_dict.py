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
def _get_body_as_dict(request):
    data = request.data
    if isinstance(data, bytes):
        data = json.loads(data.decode('utf-8'))
    elif isinstance(data, str):
        data = json.loads(data)
    return data