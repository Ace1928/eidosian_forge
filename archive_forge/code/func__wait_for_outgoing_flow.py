import enum
import logging
import time
import types
import typing
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, Semaphore, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
def _wait_for_outgoing_flow(self, request: Request, stream_id: int) -> int:
    """
        Returns the maximum allowable outgoing flow for a given stream.

        If the allowable flow is zero, then waits on the network until
        WindowUpdated frames have increased the flow rate.
        https://tools.ietf.org/html/rfc7540#section-6.9
        """
    local_flow: int = self._h2_state.local_flow_control_window(stream_id)
    max_frame_size: int = self._h2_state.max_outbound_frame_size
    flow = min(local_flow, max_frame_size)
    while flow == 0:
        self._receive_events(request)
        local_flow = self._h2_state.local_flow_control_window(stream_id)
        max_frame_size = self._h2_state.max_outbound_frame_size
        flow = min(local_flow, max_frame_size)
    return flow