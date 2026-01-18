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
def _add_base64_padding(b):
    """Add padding to base64 encoded bytes.

    pymacaroons does not give padded base64 bytes from serialization.

    @param bytes b to be padded.
    @return a padded bytes.
    """
    return b + b'=' * (-len(b) % 4)