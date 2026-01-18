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
class InvalidCdnSignedUrlError(core_exceptions.Error):
    """Invalid URL error."""