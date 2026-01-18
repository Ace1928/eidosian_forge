from __future__ import division
import collections
import functools
import itertools
import logging
import threading
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Set, Tuple
import uuid
import grpc  # type: ignore
from google.api_core import bidi
from google.api_core import exceptions
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.subscriber._protocol import dispatcher
from google.cloud.pubsub_v1.subscriber._protocol import heartbeater
from google.cloud.pubsub_v1.subscriber._protocol import histogram
from google.cloud.pubsub_v1.subscriber._protocol import leaser
from google.cloud.pubsub_v1.subscriber._protocol import messages_on_hold
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
import google.cloud.pubsub_v1.subscriber.message
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler
from google.pubsub_v1 import types as gapic_types
from google.rpc.error_details_pb2 import ErrorInfo  # type: ignore
from google.rpc import code_pb2  # type: ignore
from google.rpc import status_pb2
def _should_terminate(self, exception: BaseException) -> bool:
    """Determine if an error on the RPC stream should be terminated.

        If the exception is one of the terminating exceptions, this will signal
        to the consumer thread that it should terminate.

        This will cause the stream to exit when it returns :data:`True`.

        Returns:
            Indicates if the caller should terminate or attempt recovery.
            Will be :data:`True` if the ``exception`` is "acceptable", i.e.
            in a list of terminating exceptions.
        """
    exception = _wrap_as_exception(exception)
    if isinstance(exception, _TERMINATING_STREAM_ERRORS):
        _LOGGER.debug('Observed terminating stream error %s', exception)
        return True
    _LOGGER.debug('Observed non-terminating stream error %s', exception)
    return False