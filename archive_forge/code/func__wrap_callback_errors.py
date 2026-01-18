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
def _wrap_callback_errors(callback: Callable[['google.cloud.pubsub_v1.subscriber.message.Message'], Any], on_callback_error: Callable[[Exception], Any], message: 'google.cloud.pubsub_v1.subscriber.message.Message'):
    """Wraps a user callback so that if an exception occurs the message is
    nacked.

    Args:
        callback: The user callback.
        message: The Pub/Sub message.
    """
    try:
        callback(message)
    except Exception as exc:
        _LOGGER.exception('Top-level exception occurred in callback while processing a message')
        message.nack()
        on_callback_error(exc)