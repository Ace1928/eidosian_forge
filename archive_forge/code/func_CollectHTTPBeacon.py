from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def CollectHTTPBeacon(self, url, method, body, headers):
    """Record a custom event to an arbitrary endpoint.

    Args:
      url: str, The full url of the endpoint to hit.
      method: str, The HTTP method to issue.
      body: str, The body to send with the request.
      headers: {str: str}, A map of headers to values to include in the request.
    """
    self._metrics.append((url, method, body, headers))