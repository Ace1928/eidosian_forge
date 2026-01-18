import base64
import binascii
import ipaddress
import json
import webbrowser
from datetime import datetime
import six
from pymacaroons import Macaroon
from pymacaroons.serializers import json_serializer
import six.moves.http_cookiejar as http_cookiejar
from six.moves.urllib.parse import urlparse
def _is_ip_addr(h):
    if six.PY2:
        try:
            h = h.decode('ascii')
        except UnicodeDecodeError:
            return False
    try:
        ipaddress.ip_address(h)
    except ValueError:
        return False
    return True