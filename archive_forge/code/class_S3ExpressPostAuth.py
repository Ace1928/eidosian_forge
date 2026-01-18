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
class S3ExpressPostAuth(S3ExpressAuth):
    REQUIRES_IDENTITY_CACHE = True

    def add_auth(self, request):
        datetime_now = datetime.datetime.utcnow()
        request.context['timestamp'] = datetime_now.strftime(SIGV4_TIMESTAMP)
        fields = {}
        if request.context.get('s3-presign-post-fields', None) is not None:
            fields = request.context['s3-presign-post-fields']
        policy = {}
        conditions = []
        if request.context.get('s3-presign-post-policy', None) is not None:
            policy = request.context['s3-presign-post-policy']
            if policy.get('conditions', None) is not None:
                conditions = policy['conditions']
        policy['conditions'] = conditions
        fields['x-amz-algorithm'] = 'AWS4-HMAC-SHA256'
        fields['x-amz-credential'] = self.scope(request)
        fields['x-amz-date'] = request.context['timestamp']
        conditions.append({'x-amz-algorithm': 'AWS4-HMAC-SHA256'})
        conditions.append({'x-amz-credential': self.scope(request)})
        conditions.append({'x-amz-date': request.context['timestamp']})
        if self.credentials.token is not None:
            fields['X-Amz-S3session-Token'] = self.credentials.token
            conditions.append({'X-Amz-S3session-Token': self.credentials.token})
        fields['policy'] = base64.b64encode(json.dumps(policy).encode('utf-8')).decode('utf-8')
        fields['x-amz-signature'] = self.signature(fields['policy'], request)
        request.context['s3-presign-post-fields'] = fields
        request.context['s3-presign-post-policy'] = policy