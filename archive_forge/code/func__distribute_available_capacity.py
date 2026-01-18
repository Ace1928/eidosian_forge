from collections import OrderedDict
import logging
import threading
from typing import Dict, Optional, Type
import warnings
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.publisher import exceptions
def _distribute_available_capacity(self) -> None:
    """Distribute available capacity among the waiting threads in FIFO order.

        The method assumes that the caller has obtained ``_operational_lock``.
        """
    available_slots = self._settings.message_limit - self._message_count - self._reserved_slots
    available_bytes = self._settings.byte_limit - self._total_bytes - self._reserved_bytes
    for reservation in self._waiting.values():
        if available_slots <= 0 and available_bytes <= 0:
            break
        if available_slots > 0 and (not reservation.has_slot):
            reservation.has_slot = True
            self._reserved_slots += 1
            available_slots -= 1
        if available_bytes <= 0:
            continue
        bytes_still_needed = reservation.bytes_needed - reservation.bytes_reserved
        if bytes_still_needed < 0:
            msg = 'Too many bytes reserved: {} / {}'.format(reservation.bytes_reserved, reservation.bytes_needed)
            warnings.warn(msg, category=RuntimeWarning)
            bytes_still_needed = 0
        can_give = min(bytes_still_needed, available_bytes)
        reservation.bytes_reserved += can_give
        self._reserved_bytes += can_give
        available_bytes -= can_give