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
class S3SigV4Auth(SigV4Auth):

    def _modify_request_before_signing(self, request):
        super()._modify_request_before_signing(request)
        if 'X-Amz-Content-SHA256' in request.headers:
            del request.headers['X-Amz-Content-SHA256']
        request.headers['X-Amz-Content-SHA256'] = self.payload(request)

    def _should_sha256_sign_payload(self, request):
        client_config = request.context.get('client_config')
        s3_config = getattr(client_config, 's3', None)
        if s3_config is None:
            s3_config = {}
        sign_payload = s3_config.get('payload_signing_enabled', None)
        if sign_payload is not None:
            return sign_payload
        checksum_header = 'Content-MD5'
        checksum_context = request.context.get('checksum', {})
        algorithm = checksum_context.get('request_algorithm')
        if isinstance(algorithm, dict) and algorithm.get('in') == 'header':
            checksum_header = algorithm['name']
        if not request.url.startswith('https') or checksum_header not in request.headers:
            return True
        if request.context.get('has_streaming_input', False):
            return False
        return super()._should_sha256_sign_payload(request)

    def _normalize_url_path(self, path):
        return path