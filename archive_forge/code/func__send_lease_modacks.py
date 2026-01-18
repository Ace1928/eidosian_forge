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
def _send_lease_modacks(self, ack_ids: Iterable[str], ack_deadline: float, warn_on_invalid=True) -> Set[str]:
    exactly_once_enabled = False
    with self._exactly_once_enabled_lock:
        exactly_once_enabled = self._exactly_once_enabled
    if exactly_once_enabled:
        items = [requests.ModAckRequest(ack_id, ack_deadline, futures.Future()) for ack_id in ack_ids]
        assert self._dispatcher is not None
        self._dispatcher.modify_ack_deadline(items, ack_deadline)
        expired_ack_ids = set()
        for req in items:
            try:
                assert req.future is not None
                req.future.result()
            except AcknowledgeError as ack_error:
                if ack_error.error_code != AcknowledgeStatus.INVALID_ACK_ID or warn_on_invalid:
                    _LOGGER.warning('AcknowledgeError when lease-modacking a message.', exc_info=True)
                if ack_error.error_code == AcknowledgeStatus.INVALID_ACK_ID:
                    expired_ack_ids.add(req.ack_id)
        return expired_ack_ids
    else:
        items = [requests.ModAckRequest(ack_id, self.ack_deadline, None) for ack_id in ack_ids]
        assert self._dispatcher is not None
        self._dispatcher.modify_ack_deadline(items, ack_deadline)
        return set()