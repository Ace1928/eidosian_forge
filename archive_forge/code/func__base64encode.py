import six
import base64
import sys
from pyu2f import errors
from pyu2f import u2f
from pyu2f.convenience import baseauthenticator
def _base64encode(self, bytes_data):
    """Helper method to base64 encode and return str result."""
    return base64.urlsafe_b64encode(bytes_data).decode()