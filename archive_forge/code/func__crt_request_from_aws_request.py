import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
def _crt_request_from_aws_request(self, aws_request):
    url_parts = urlsplit(aws_request.url)
    crt_path = url_parts.path if url_parts.path else '/'
    if aws_request.params:
        array = []
        for param, value in aws_request.params.items():
            value = str(value)
            array.append(f'{param}={value}')
        crt_path = crt_path + '?' + '&'.join(array)
    elif url_parts.query:
        crt_path = f'{crt_path}?{url_parts.query}'
    crt_headers = awscrt.http.HttpHeaders(aws_request.headers.items())
    crt_body_stream = None
    if aws_request.body:
        if hasattr(aws_request.body, 'seek'):
            crt_body_stream = aws_request.body
        else:
            crt_body_stream = BytesIO(aws_request.body)
    crt_request = awscrt.http.HttpRequest(method=aws_request.method, path=crt_path, headers=crt_headers, body_stream=crt_body_stream)
    return crt_request