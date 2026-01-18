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
class LoggingInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor, grpc.StreamUnaryClientInterceptor, grpc.StreamStreamClientInterceptor):
    """Logging Interceptor for logging requests and responses.

  Logging is enabled if the --log-http flag is provided on any command.
  """

    def __init__(self, credentials):
        self._credentials = credentials

    def log_metadata(self, metadata):
        """Logs the metadata.

    Args:
      metadata: `metadata` to be transmitted to
        the service-side of the RPC.
    """
        redact_token = properties.VALUES.core.log_http_redact_token.GetBool()
        for h, v in sorted(metadata or [], key=lambda x: x[0]):
            if redact_token and h.lower() == IAM_AUTHORIZATION_TOKEN_HEADER:
                v = '--- Token Redacted ---'
            log.status.Print('{0}: {1}'.format(h, v))

    def log_request(self, client_call_details, request):
        """Logs information about the request.

    Args:
        client_call_details: a grpc._interceptor._ClientCallDetails
            instance containing request metadata.
        request: the request value for the RPC.
    """
        redact_token = properties.VALUES.core.log_http_redact_token.GetBool()
        log.status.Print('=======================')
        log.status.Print('==== request start ====')
        log.status.Print('method: {}'.format(client_call_details.method))
        log.status.Print('== headers start ==')
        if self._credentials:
            if redact_token:
                log.status.Print('authorization: --- Token Redacted ---')
            else:
                log.status.Print('authorization: {}'.format(self._credentials.token))
        self.log_metadata(client_call_details.metadata)
        log.status.Print('== headers end ==')
        log.status.Print('== body start ==')
        log.status.Print('{}'.format(request))
        log.status.Print('== body end ==')
        log.status.Print('==== request end ====')

    def log_response(self, response, time_taken):
        """Logs information about the request.

    Args:
        response: A grpc.Call/grpc.Future instance representing a service
            response.
        time_taken: time, in seconds, it took for the RPC to complete.
    """
        log.status.Print('---- response start ----')
        log.status.Print('code: {}'.format(response.code()))
        log.status.Print('-- headers start --')
        log.status.Print('details: {}'.format(response.details()))
        log.status.Print('-- initial metadata --')
        self.log_metadata(response.initial_metadata())
        log.status.Print('-- trailing metadata --')
        self.log_metadata(response.trailing_metadata())
        log.status.Print('-- headers end --')
        log.status.Print('-- body start --')
        log.status.Print('{}'.format(response.result()))
        log.status.Print('-- body end --')
        log.status.Print('total round trip time (request+response): {0:.3f} secs'.format(time_taken))
        log.status.Print('---- response end ----')
        log.status.Print('----------------------')

    def log_requests(self, client_call_details, request_iterator):
        for request in request_iterator:
            self.log_request(client_call_details, request)
            yield request

    def log_streaming_response(self, responses, response, time_taken):
        """Logs information about the response.

    Args:
        responses: A grpc.Call/grpc.Future instance representing a service
            response.
        response: response to log.
        time_taken: time, in seconds, it took for the RPC to complete.
    """
        log.status.Print('---- response start ----')
        log.status.Print('-- headers start --')
        log.status.Print('-- initial metadata --')
        self.log_metadata(responses.initial_metadata())
        log.status.Print('-- headers end --')
        log.status.Print('-- body start --')
        log.status.Print('{}'.format(response))
        log.status.Print('-- body end --')
        log.status.Print('total time (response): {0:.3f} secs'.format(time_taken))
        log.status.Print('---- response end ----')
        log.status.Print('----------------------')

    def log_responses(self, responses):

        def OnDone(response):
            log.status.Print('---- response start ----')
            log.status.Print('code: {}'.format(response.code()))
            log.status.Print('-- headers start --')
            log.status.Print('details: {}'.format(response.details()))
            log.status.Print('-- trailing metadata --')
            self.log_metadata(response.trailing_metadata())
            log.status.Print('-- headers end --')
            log.status.Print('---- response end ----')
            log.status.Print('----------------------')

        def LogResponse(result_generator_func):
            start_time = time.time()
            response = result_generator_func()
            time_taken = time.time() - start_time
            self.log_streaming_response(responses, response, time_taken)
            return response
        responses.add_done_callback(OnDone)
        return WrappedStreamingResponse(responses, LogResponse)

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercepts and logs API interactions.

    Overrides abstract method defined in grpc.UnaryUnaryClientInterceptor.
    Args:
        continuation: a function to continue the request process.
        client_call_details: a grpc._interceptor._ClientCallDetails
            instance containing request metadata.
        request: the request value for the RPC.
    Returns:
        A grpc.Call/grpc.Future instance representing a service response.
    """
        self.log_request(client_call_details, request)
        start_time = time.time()
        response = continuation(client_call_details, request)
        time_taken = time.time() - start_time
        self.log_response(response, time_taken)
        return response

    def intercept_unary_stream(self, continuation, client_call_details, request):
        """Intercepts a unary-stream invocation."""
        self.log_request(client_call_details, request)
        response = continuation(client_call_details, request)
        return self.log_responses(response)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        """Intercepts a stream-unary invocation asynchronously."""
        start_time = time.time()
        response = continuation(client_call_details, self.log_requests(client_call_details, request_iterator))
        time_taken = time.time() - start_time
        self.log_response(response, time_taken)
        return response

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        """Intercepts a stream-stream invocation."""
        response = continuation(client_call_details, self.log_requests(client_call_details, request_iterator))
        return self.log_responses(response)