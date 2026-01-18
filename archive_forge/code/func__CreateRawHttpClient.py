from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import google_auth_httplib2
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import encoding
import httplib2
import six
def _CreateRawHttpClient(timeout='unset', ca_certs=None):
    """Create an HTTP client matching the appropriate gcloud properties."""
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
    return HttpClient(timeout=effective_timeout, proxy_info=http_proxy.GetHttpProxyInfo(), ca_certs=ca_certs, disable_ssl_certificate_validation=no_validate)