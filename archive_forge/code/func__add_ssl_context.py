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
def _add_ssl_context(self, kwargs):
    if not self._cert_info:
        return
    context = CreateSSLContext()
    cert_chain_kwargs = {}
    if self._cert_info.keyfile:
        cert_chain_kwargs['keyfile'] = self._cert_info.keyfile
    if self._cert_info.password:
        cert_chain_kwargs['password'] = self._cert_info.password
    context.load_cert_chain(self._cert_info.certfile, **cert_chain_kwargs)
    kwargs['ssl_context'] = context