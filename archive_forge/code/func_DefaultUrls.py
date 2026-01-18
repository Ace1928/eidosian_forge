from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import socket
import ssl
from googlecloudsdk.core import config
from googlecloudsdk.core import http
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core.diagnostics import check_base
from googlecloudsdk.core.diagnostics import diagnostic_base
from googlecloudsdk.core.diagnostics import http_proxy_setup
import httplib2
import requests
from six.moves import http_client
from six.moves import urllib
import socks
def DefaultUrls():
    """Returns a list of hosts whose reachability is essential for the Cloud SDK.

  Returns:
    A list of urls (str) to check reachability for.
  """
    urls = ['https://accounts.google.com', 'https://cloudresourcemanager.googleapis.com/v1beta1/projects', 'https://www.googleapis.com/auth/cloud-platform']
    download_urls = properties.VALUES.component_manager.snapshot_url.Get() or config.INSTALLATION_CONFIG.snapshot_url
    urls.extend((u for u in download_urls.split(',') if urllib.parse.urlparse(u).scheme in ('http', 'https')))
    return urls