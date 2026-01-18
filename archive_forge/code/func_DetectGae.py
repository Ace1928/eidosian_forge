import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def DetectGae():
    """Determine whether or not we're running on GAE.

    This is based on:
      https://developers.google.com/appengine/docs/python/#The_Environment

    Returns:
      True iff we're running on GAE.
    """
    server_software = os.environ.get('SERVER_SOFTWARE', '')
    return server_software.startswith('Development/') or server_software.startswith('Google App Engine/')