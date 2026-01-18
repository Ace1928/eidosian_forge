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
class S3ExpressQueryAuth(S3ExpressAuth):
    DEFAULT_EXPIRES = 300
    REQUIRES_IDENTITY_CACHE = True

    def __init__(self, credentials, service_name, region_name, *, identity_cache, expires=DEFAULT_EXPIRES):
        super().__init__(credentials, service_name, region_name, identity_cache=identity_cache)
        self._expires = expires

    def _modify_request_before_signing(self, request):
        content_type = request.headers.get('content-type')
        blocklisted_content_type = 'application/x-www-form-urlencoded; charset=utf-8'
        if content_type == blocklisted_content_type:
            del request.headers['content-type']
        signed_headers = self.signed_headers(self.headers_to_sign(request))
        auth_params = {'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': self.scope(request), 'X-Amz-Date': request.context['timestamp'], 'X-Amz-Expires': self._expires, 'X-Amz-SignedHeaders': signed_headers}
        if self.credentials.token is not None:
            auth_params['X-Amz-S3session-Token'] = self.credentials.token
        url_parts = urlsplit(request.url)
        query_string_parts = parse_qs(url_parts.query, keep_blank_values=True)
        query_dict = {k: v[0] for k, v in query_string_parts.items()}
        if request.params:
            query_dict.update(request.params)
            request.params = {}
        operation_params = ''
        if request.data:
            query_dict.update(_get_body_as_dict(request))
            request.data = ''
        if query_dict:
            operation_params = percent_encode_sequence(query_dict) + '&'
        new_query_string = f'{operation_params}{percent_encode_sequence(auth_params)}'
        p = url_parts
        new_url_parts = (p[0], p[1], p[2], new_query_string, p[4])
        request.url = urlunsplit(new_url_parts)

    def _inject_signature_to_request(self, request, signature):
        request.url += '&X-Amz-Signature=%s' % signature

    def _normalize_url_path(self, path):
        return path

    def payload(self, request):
        return UNSIGNED_PAYLOAD