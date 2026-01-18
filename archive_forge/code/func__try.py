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
def _try(fmt):
    try:
        return _day0.strftime(fmt) == '0001'
    except ValueError:
        return False