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
class ClientCallDetailsInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor, grpc.StreamUnaryClientInterceptor, grpc.StreamStreamClientInterceptor):
    """Generic Client Interceptor that modifies the ClientCallDetails."""

    def __init__(self, fn):
        self._fn = fn

    def intercept_call(self, continuation, client_call_details, request):
        """Intercepts a RPC.

    Args:
      continuation: A function that proceeds with the invocation by
        executing the next interceptor in chain or invoking the
        actual RPC on the underlying Channel. It is the interceptor's
        responsibility to call it if it decides to move the RPC forward.
        The interceptor can use
        `response_future = continuation(client_call_details, request)`
        to continue with the RPC.
      client_call_details: A ClientCallDetails object describing the
        outgoing RPC.
      request: The request value for the RPC.

    Returns:
        If the response is unary:
          An object that is both a Call for the RPC and a Future.
          In the event of RPC completion, the return Call-Future's
          result value will be the response message of the RPC.
          Should the event terminate with non-OK status, the returned
          Call-Future's exception value will be an RpcError.

        If the response is streaming:
          An object that is both a Call for the RPC and an iterator of
          response values. Drawing response values from the returned
          Call-iterator may raise RpcError indicating termination of
          the RPC with non-OK status.
    """
        new_details = self._fn(client_call_details)
        return continuation(new_details, request)

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercepts a unary-unary invocation asynchronously."""
        return self.intercept_call(continuation, client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        """Intercepts a unary-stream invocation."""
        return self.intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        """Intercepts a stream-unary invocation asynchronously."""
        return self.intercept_call(continuation, client_call_details, request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        """Intercepts a stream-stream invocation."""
        return self.intercept_call(continuation, client_call_details, request_iterator)