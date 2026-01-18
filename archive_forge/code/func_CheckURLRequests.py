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
def CheckURLRequests(url):
    try:
        core_requests.GetSession(timeout=_NETWORK_TIMEOUT).request('GET', url)
    except requests.exceptions.RequestException as err:
        msg = 'requests cannot reach {0}:\n{1}\n'.format(url, err)
        return check_base.Failure(message=msg, exception=err)