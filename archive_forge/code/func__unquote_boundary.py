import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def _unquote_boundary(self, b):
    return b[:2] + email_utils.unquote(b[2:-2].decode('ascii')).encode('ascii') + b[-2:]