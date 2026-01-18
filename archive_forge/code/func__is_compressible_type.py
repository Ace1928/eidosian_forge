import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
def _is_compressible_type(request_dict):
    body = request_dict['body']
    if isinstance(body, dict):
        body = urlencode(body, doseq=True, encoding='utf-8').encode('utf-8')
        request_dict['body'] = body
    is_supported_type = isinstance(body, (str, bytes, bytearray))
    return is_supported_type or hasattr(body, 'read')