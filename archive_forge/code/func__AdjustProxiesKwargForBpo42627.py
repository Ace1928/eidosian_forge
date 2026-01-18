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
def _AdjustProxiesKwargForBpo42627(gcloud_proxy_info, environment_proxies, orig_request_method, *args, **kwargs):
    """Returns proxies to workaround https://bugs.python.org/issue42627 if needed.

  Args:
    gcloud_proxy_info: str, Proxy info from gcloud properties.
    environment_proxies: dict, Proxy config from http/https_proxy env vars.
    orig_request_method: function, The original requests.Session.request method.
    *args: Positional arguments to the original request method.
    **kwargs: Keyword arguments to the original request method.
  Returns:
    Optional[dict], Adjusted proxies to pass to the request method, or None if
      no adjustment is necessary.
  """
    if gcloud_proxy_info or environment_proxies:
        return None
    url = inspect.getcallargs(orig_request_method, *args, **kwargs)['url']
    proxies = requests.utils.get_environ_proxies(url)
    https_proxy = proxies.get('https')
    if not https_proxy:
        return None
    if not https_proxy.startswith('https://'):
        return None
    return {'https': https_proxy.replace('https://', 'http://', 1)}