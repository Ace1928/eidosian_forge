import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
def _set_compression_header(headers, encoding):
    ce_header = headers.get('Content-Encoding')
    if ce_header is None:
        headers['Content-Encoding'] = encoding
    else:
        headers['Content-Encoding'] = f'{ce_header},{encoding}'