import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
def _gzip_compress_body(body):
    if isinstance(body, str):
        return gzip_compress(body.encode('utf-8'))
    elif isinstance(body, (bytes, bytearray)):
        return gzip_compress(body)
    elif hasattr(body, 'read'):
        if hasattr(body, 'seek') and hasattr(body, 'tell'):
            current_position = body.tell()
            compressed_obj = _gzip_compress_fileobj(body)
            body.seek(current_position)
            return compressed_obj
        return _gzip_compress_fileobj(body)