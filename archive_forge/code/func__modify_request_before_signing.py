import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
def _modify_request_before_signing(self, request):
    super()._modify_request_before_signing(request)
    content_type = request.headers.get('content-type')
    if content_type == 'application/x-www-form-urlencoded; charset=utf-8':
        del request.headers['content-type']
    url_parts = urlsplit(request.url)
    query_dict = {k: v[0] for k, v in parse_qs(url_parts.query, keep_blank_values=True).items()}
    if request.params:
        query_dict.update(request.params)
        request.params = {}
    if request.data:
        query_dict.update(_get_body_as_dict(request))
        request.data = ''
    new_query_string = percent_encode_sequence(query_dict)
    p = url_parts
    new_url_parts = (p[0], p[1], p[2], new_query_string, p[4])
    request.url = urlunsplit(new_url_parts)