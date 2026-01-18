import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
Attempt to compress the request body using the modeled encodings.