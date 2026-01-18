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
def _cleanup_sender(self, sender_ref: annotatable_weakref) -> None:
    """Disconnect all receivers from a sender."""
    sender_id = cast(IdentityType, sender_ref.sender_id)
    assert sender_id != ANY_ID
    self._weak_senders.pop(sender_id, None)
    for receiver_id in self._by_sender.pop(sender_id, ()):
        self._by_receiver[receiver_id].discard(sender_id)