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
def _set_necessary_date_headers(self, request):
    if 'Date' in request.headers:
        del request.headers['Date']
        datetime_timestamp = datetime.datetime.strptime(request.context['timestamp'], SIGV4_TIMESTAMP)
        request.headers['Date'] = formatdate(int(calendar.timegm(datetime_timestamp.timetuple())))
        if 'X-Amz-Date' in request.headers:
            del request.headers['X-Amz-Date']
    else:
        if 'X-Amz-Date' in request.headers:
            del request.headers['X-Amz-Date']
        request.headers['X-Amz-Date'] = request.context['timestamp']