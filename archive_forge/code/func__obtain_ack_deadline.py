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
def _obtain_ack_deadline(self, maybe_update: bool) -> float:
    """The actual `ack_deadline` implementation.

        This method is "sticky". It will only perform the computations to check on the
        right ACK deadline if explicitly requested AND if the histogram with past
        time-to-ack data has gained a significant amount of new information.

        Args:
            maybe_update:
                If ``True``, also update the current ACK deadline before returning it if
                enough new ACK data has been gathered.

        Returns:
            The current ACK deadline in seconds to use.
        """
    with self._ack_deadline_lock:
        if not maybe_update:
            return self._ack_deadline
        target_size = min(self._last_histogram_size * 2, self._last_histogram_size + 100)
        hist_size = len(self.ack_histogram)
        if hist_size > target_size:
            self._last_histogram_size = hist_size
            self._ack_deadline = self.ack_histogram.percentile(percent=99)
        if self.flow_control.max_duration_per_lease_extension > 0:
            flow_control_setting = max(self.flow_control.max_duration_per_lease_extension, histogram.MIN_ACK_DEADLINE)
            self._ack_deadline = min(self._ack_deadline, flow_control_setting)
        if self.flow_control.min_duration_per_lease_extension > 0:
            flow_control_setting = min(self.flow_control.min_duration_per_lease_extension, histogram.MAX_ACK_DEADLINE)
            self._ack_deadline = max(self._ack_deadline, flow_control_setting)
        elif self._exactly_once_enabled:
            self._ack_deadline = max(self._ack_deadline, _MIN_ACK_DEADLINE_SECS_WHEN_EXACTLY_ONCE_ENABLED)
        if self._ack_deadline > self._stream_ack_deadline:
            self._stream_ack_deadline = self._ack_deadline
        return self._ack_deadline