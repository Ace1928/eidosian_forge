from google.auth.transport.requests import AuthorizedSession  # type: ignore
import json  # type: ignore
import grpc  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import path_template
from google.api_core import gapic_v1
from cloudsdk.google.protobuf import json_format
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging
from .base import LoggingServiceV2Transport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class _TailLogEntries(LoggingServiceV2RestStub):

    def __hash__(self):
        return hash('TailLogEntries')

    def __call__(self, request: logging.TailLogEntriesRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> rest_streaming.ResponseIterator:
        raise NotImplementedError('Method TailLogEntries is not available over REST transport')