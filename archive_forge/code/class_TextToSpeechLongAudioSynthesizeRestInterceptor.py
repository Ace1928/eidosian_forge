import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import (
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts_lrs
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import TextToSpeechLongAudioSynthesizeTransport
class TextToSpeechLongAudioSynthesizeRestInterceptor:
    """Interceptor for TextToSpeechLongAudioSynthesize.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the TextToSpeechLongAudioSynthesizeRestTransport.

    .. code-block:: python
        class MyCustomTextToSpeechLongAudioSynthesizeInterceptor(TextToSpeechLongAudioSynthesizeRestInterceptor):
            def pre_synthesize_long_audio(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_synthesize_long_audio(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = TextToSpeechLongAudioSynthesizeRestTransport(interceptor=MyCustomTextToSpeechLongAudioSynthesizeInterceptor())
        client = TextToSpeechLongAudioSynthesizeClient(transport=transport)


    """

    def pre_synthesize_long_audio(self, request: cloud_tts_lrs.SynthesizeLongAudioRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_tts_lrs.SynthesizeLongAudioRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for synthesize_long_audio

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TextToSpeechLongAudioSynthesize server.
        """
        return (request, metadata)

    def post_synthesize_long_audio(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for synthesize_long_audio

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeechLongAudioSynthesize server but before
        it is returned to user code.
        """
        return response

    def pre_get_operation(self, request: operations_pb2.GetOperationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[operations_pb2.GetOperationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TextToSpeechLongAudioSynthesize server.
        """
        return (request, metadata)

    def post_get_operation(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeechLongAudioSynthesize server but before
        it is returned to user code.
        """
        return response

    def pre_list_operations(self, request: operations_pb2.ListOperationsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[operations_pb2.ListOperationsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TextToSpeechLongAudioSynthesize server.
        """
        return (request, metadata)

    def post_list_operations(self, response: operations_pb2.ListOperationsResponse) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeechLongAudioSynthesize server but before
        it is returned to user code.
        """
        return response