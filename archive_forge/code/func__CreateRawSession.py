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
def _CreateRawSession(timeout='unset', ca_certs=None, session=None, client_certificate=None, client_key=None):
    """Create a requests.Session matching the appropriate gcloud properties."""
    if timeout != 'unset':
        effective_timeout = timeout
    else:
        effective_timeout = transport.GetDefaultTimeout()
    no_validate = properties.VALUES.auth.disable_ssl_validation.GetBool() or False
    ca_certs_property = properties.VALUES.core.custom_ca_certs_file.Get()
    if ca_certs_property:
        ca_certs = ca_certs_property
    if no_validate:
        ca_certs = None
    return Session(timeout=effective_timeout, ca_certs=ca_certs, disable_ssl_certificate_validation=no_validate, session=session, client_certificate=client_certificate, client_key=client_key)