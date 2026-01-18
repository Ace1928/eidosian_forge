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
def _schedule_message_on_hold(self, msg: 'google.cloud.pubsub_v1.subscriber.message.Message'):
    """Schedule a message on hold to be sent to the user and change on-hold-bytes.

        The method assumes the caller has acquired the ``_pause_resume_lock``.

        Args:
            msg: The message to schedule to be sent to the user.
        """
    assert msg, 'Message must not be None.'
    self._on_hold_bytes -= msg.size
    if self._on_hold_bytes < 0:
        _LOGGER.warning('On hold bytes was unexpectedly negative: %s', self._on_hold_bytes)
        self._on_hold_bytes = 0
    _LOGGER.debug('Released held message, scheduling callback for it, still on hold %s (bytes %s).', self._messages_on_hold.size, self._on_hold_bytes)
    assert self._scheduler is not None
    assert self._callback is not None
    self._scheduler.schedule(self._callback, msg)