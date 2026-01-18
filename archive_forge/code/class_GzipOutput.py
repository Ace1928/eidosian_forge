import gzip
import io
from paste.response import header_value, remove_header
from paste.httpheaders import CONTENT_LENGTH
class GzipOutput(object):
    pass