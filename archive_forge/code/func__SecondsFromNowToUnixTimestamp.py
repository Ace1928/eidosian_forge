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
def _SecondsFromNowToUnixTimestamp(seconds_from_now):
    """Converts the number of seconds from now into a unix timestamp."""
    return int(time.time() + seconds_from_now)