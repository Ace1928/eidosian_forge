from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from google.api_core import bidi
from google.rpc import error_details_pb2
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.calliope import base
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_proxy_types
import grpc
from six.moves import urllib
import socks
def MakeProxyFromProperties():
    """Returns the proxy string for use by grpc from gcloud properties."""
    proxy_type = properties.VALUES.proxy.proxy_type.Get()
    proxy_address = properties.VALUES.proxy.address.Get()
    proxy_port = properties.VALUES.proxy.port.GetInt()
    proxy_prop_set = len([f for f in (proxy_type, proxy_address, proxy_port) if f])
    if proxy_prop_set > 0 and proxy_prop_set != 3:
        raise properties.InvalidValueError('Please set all or none of the following properties: proxy/type, proxy/address and proxy/port')
    if not proxy_prop_set:
        return
    proxy_user = properties.VALUES.proxy.username.Get()
    proxy_pass = properties.VALUES.proxy.password.Get()
    http_proxy_type = http_proxy_types.PROXY_TYPE_MAP[proxy_type]
    if http_proxy_type != socks.PROXY_TYPE_HTTP:
        raise ValueError('Unsupported proxy type for gRPC: {}'.format(proxy_type))
    if proxy_user or proxy_pass:
        proxy_auth = ':'.join((urllib.parse.quote(x) or '' for x in (proxy_user, proxy_pass)))
        proxy_auth += '@'
    else:
        proxy_auth = ''
    return 'http://{}{}:{}'.format(proxy_auth, proxy_address, proxy_port)