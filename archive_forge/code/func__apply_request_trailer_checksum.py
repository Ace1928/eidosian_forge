import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def _apply_request_trailer_checksum(request):
    checksum_context = request.get('context', {}).get('checksum', {})
    algorithm = checksum_context.get('request_algorithm')
    location_name = algorithm['name']
    checksum_cls = _CHECKSUM_CLS.get(algorithm['algorithm'])
    headers = request['headers']
    body = request['body']
    if location_name in headers:
        return
    headers['Transfer-Encoding'] = 'chunked'
    if 'Content-Encoding' in headers:
        headers['Content-Encoding'] += ',aws-chunked'
    else:
        headers['Content-Encoding'] = 'aws-chunked'
    headers['X-Amz-Trailer'] = location_name
    content_length = determine_content_length(body)
    if content_length is not None:
        headers['X-Amz-Decoded-Content-Length'] = str(content_length)
    if isinstance(body, (bytes, bytearray)):
        body = io.BytesIO(body)
    request['body'] = AwsChunkedWrapper(body, checksum_cls=checksum_cls, checksum_name=location_name)