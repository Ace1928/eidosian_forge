from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
class _CallableStub(object):
    """Stub for the grpc.*MultiCallable interfaces."""

    def __init__(self, method, channel):
        self._method = method
        self._channel = channel
        self.response = None
        "Union[protobuf.Message, Callable[protobuf.Message], exception]:\n        The response to give when invoking this callable. If this is a\n        callable, it will be invoked with the request protobuf. If it's an\n        exception, the exception will be raised when this is invoked.\n        "
        self.responses = None
        'Iterator[\n            Union[protobuf.Message, Callable[protobuf.Message], exception]]:\n        An iterator of responses. If specified, self.response will be populated\n        on each invocation by calling ``next(self.responses)``.'
        self.requests = []
        'List[protobuf.Message]: All requests sent to this callable.'
        self.calls = []
        'List[Tuple]: All invocations of this callable. Each tuple is the\n        request, timeout, metadata, compression, and credentials.'

    def __call__(self, request, timeout=None, metadata=None, credentials=None, compression=None):
        self._channel.requests.append(_ChannelRequest(self._method, request))
        self.calls.append(_MethodCall(request, timeout, metadata, credentials, compression))
        self.requests.append(request)
        response = self.response
        if self.responses is not None:
            if response is None:
                response = next(self.responses)
            else:
                raise ValueError('{method}.response and {method}.responses are mutually exclusive.'.format(method=self._method))
        if callable(response):
            return response(request)
        if isinstance(response, Exception):
            raise response
        if response is not None:
            return response
        raise ValueError('Method stub for "{}" has no response.'.format(self._method))