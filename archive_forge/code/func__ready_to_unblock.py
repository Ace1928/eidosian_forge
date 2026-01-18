from collections import OrderedDict
import logging
import threading
from typing import Dict, Optional, Type
import warnings
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.publisher import exceptions
def _ready_to_unblock(self) -> bool:
    """Determine if any of the threads waiting to add a message can proceed.

        The method assumes that the caller has obtained ``_operational_lock``.
        """
    if self._waiting:
        first_reservation = next(iter(self._waiting.values()))
        return first_reservation.bytes_reserved >= first_reservation.bytes_needed and first_reservation.has_slot
    return False