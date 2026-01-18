from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import inspect
import io
from google.auth.transport import requests as google_auth_requests
from google.auth.transport.requests import _MutualTlsOffloadAdapter
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import httplib2
import requests
import six
from six.moves import http_client as httplib
from six.moves import urllib
import socks
from urllib3.util.ssl_ import create_urllib3_context
def _HasBpo42627():
    """Returns whether Python is affected by https://bugs.python.org/issue42627.

  Due to a bug in Python's standard library, urllib.request misparses the
  Windows registry proxy settings and assumes that HTTPS URLs should use an
  HTTPS proxy, when in fact they should use an HTTP proxy.

  This bug affects PY<3.9, as well as lower patch versions of 3.9, 3.10, and
  3.11.

  Returns:
    True if proxies read from the Windows registry are being parsed incorrectly.
  """
    return platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS and hasattr(urllib_request, 'getproxies_registry') and urllib_request.getproxies_registry().get('https', '').startswith('https://')