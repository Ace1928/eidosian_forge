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
def _wrap_as_exception(maybe_exception: Any) -> BaseException:
    """Wrap an object as a Python exception, if needed.

    Args:
        maybe_exception: The object to wrap, usually a gRPC exception class.

    Returns:
         The argument itself if an instance of ``BaseException``, otherwise
         the argument represented as an instance of ``Exception`` (sub)class.
    """
    if isinstance(maybe_exception, grpc.RpcError):
        return exceptions.from_grpc_error(maybe_exception)
    elif isinstance(maybe_exception, BaseException):
        return maybe_exception
    return Exception(maybe_exception)