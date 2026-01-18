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
def _maybe_release_messages(self) -> None:
    """Release (some of) the held messages if the current load allows for it.

        The method tries to release as many messages as the current leaser load
        would allow. Each released message is added to the lease management,
        and the user callback is scheduled for it.

        If there are currently no messages on hold, or if the leaser is
        already overloaded, this method is effectively a no-op.

        The method assumes the caller has acquired the ``_pause_resume_lock``.
        """
    released_ack_ids = []
    while self.load < _MAX_LOAD:
        msg = self._messages_on_hold.get()
        if not msg:
            break
        self._schedule_message_on_hold(msg)
        released_ack_ids.append(msg.ack_id)
    assert self._leaser is not None
    self._leaser.start_lease_expiry_timer(released_ack_ids)