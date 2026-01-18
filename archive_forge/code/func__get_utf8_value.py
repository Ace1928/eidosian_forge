import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
@staticmethod
def _get_utf8_value(value):
    """Get the UTF8-encoded version of a value."""
    if not isinstance(value, (str, bytes)):
        value = str(value)
    if isinstance(value, str):
        return value.encode('utf-8')
    else:
        return value