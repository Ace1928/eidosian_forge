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
def ConstructMessageFromFailures(failures, first_run):
    """Constructs error messages along with diagnostic information."""
    message = 'Reachability Check {0}.\n'.format('failed' if first_run else 'still does not pass')
    for failure in failures:
        message += '    {0}\n'.format(failure.message)
    if first_run:
        message += 'Network connection problems may be due to proxy or firewall settings.\n'
    return message