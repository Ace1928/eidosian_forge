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
def _inject_signature(self, request, signature):
    query_dict = {}
    query_dict['AWSAccessKeyId'] = self.credentials.access_key
    query_dict['Signature'] = signature
    for header_key in request.headers:
        lk = header_key.lower()
        if header_key == 'Date':
            query_dict['Expires'] = request.headers['Date']
        elif lk.startswith('x-amz-') or lk in ('content-md5', 'content-type'):
            query_dict[lk] = request.headers[lk]
    new_query_string = percent_encode_sequence(query_dict)
    p = urlsplit(request.url)
    if p[3]:
        new_query_string = f'{p[3]}&{new_query_string}'
    new_url_parts = (p[0], p[1], p[2], new_query_string, p[4])
    request.url = urlunsplit(new_url_parts)