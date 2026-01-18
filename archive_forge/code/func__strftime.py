import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
def _strftime(value):
    if isinstance(value, datetime):
        return _iso8601_format(value)
    if not isinstance(value, (tuple, time.struct_time)):
        if value == 0:
            value = time.time()
        value = time.localtime(value)
    return '%04d%02d%02dT%02d:%02d:%02d' % value[:6]