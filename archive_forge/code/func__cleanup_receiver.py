from __future__ import annotations
import typing as t
from collections import defaultdict
from contextlib import contextmanager
from inspect import iscoroutinefunction
from warnings import warn
from weakref import WeakValueDictionary
from blinker._utilities import annotatable_weakref
from blinker._utilities import hashable_identity
from blinker._utilities import IdentityType
from blinker._utilities import lazy_property
from blinker._utilities import reference
from blinker._utilities import symbol
from blinker._utilities import WeakTypes
def _cleanup_receiver(self, receiver_ref: annotatable_weakref) -> None:
    """Disconnect a receiver from all senders."""
    self._disconnect(cast(IdentityType, receiver_ref.receiver_id), ANY_ID)