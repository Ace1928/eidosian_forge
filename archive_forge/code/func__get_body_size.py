import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
def _get_body_size(body):
    size = determine_content_length(body)
    if size is None:
        logger.debug('Unable to get length of the request body: %s. Skipping compression.', body)
        size = 0
    return size