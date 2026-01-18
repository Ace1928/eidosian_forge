from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import hmac
import time
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core.util import encoding
import six.moves.urllib.parse
def _GetSignature(key, url):
    """Gets the base64url encoded HMAC-SHA1 signature of the specified URL.

  Args:
    key: The key value to use for signing.
    url: The url to use for signing.

  Returns:
    The signature of the specified URL calculated using HMAC-SHA1 signature
    digest and encoding the result using base64url.
  """
    signature = hmac.new(key, url, hashlib.sha1).digest()
    return base64.urlsafe_b64encode(signature)