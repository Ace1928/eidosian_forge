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
def MakeTransport(client_class, credentials, address_override_func, mtls_enabled=False, attempt_direct_path=False):
    """Instantiates a grpc transport."""
    transport_class = client_class.get_transport_class()
    address = _GetAddress(client_class, address_override_func, mtls_enabled)
    channel = transport_class.create_channel(host=address, credentials=credentials, ssl_credentials=GetSSLCredentials(mtls_enabled), options=MakeChannelOptions(), attempt_direct_path=attempt_direct_path)
    interceptors = []
    interceptors.append(RequestReasonInterceptor())
    interceptors.append(UserAgentInterceptor())
    interceptors.append(TimeoutInterceptor())
    interceptors.append(IAMAuthHeadersInterceptor())
    interceptors.append(RPCDurationReporterInterceptor())
    interceptors.append(QuotaProjectInterceptor(credentials))
    interceptors.append(APIEnablementInterceptor())
    interceptors.append(RequestOrgRestrictionInterceptor())
    if properties.VALUES.core.log_http.GetBool():
        interceptors.append(LoggingInterceptor(credentials))
    channel = grpc.intercept_channel(channel, *interceptors)
    return transport_class(channel=channel, host=address)