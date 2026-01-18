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
def _cleanup_bookkeeping(self) -> None:
    """Prune unused sender/receiver bookkeeping. Not threadsafe.

        Connecting & disconnecting leave behind a small amount of bookkeeping
        for the receiver and sender values. Typical workloads using Blinker,
        for example in most web apps, Flask, CLI scripts, etc., are not
        adversely affected by this bookkeeping.

        With a long-running Python process performing dynamic signal routing
        with high volume- e.g. connecting to function closures, "senders" are
        all unique object instances, and doing all of this over and over- you
        may see memory usage will grow due to extraneous bookkeeping. (An empty
        set() for each stale sender/receiver pair.)

        This method will prune that bookkeeping away, with the caveat that such
        pruning is not threadsafe. The risk is that cleanup of a fully
        disconnected receiver/sender pair occurs while another thread is
        connecting that same pair. If you are in the highly dynamic, unique
        receiver/sender situation that has lead you to this method, that
        failure mode is perhaps not a big deal for you.
        """
    for mapping in (self._by_sender, self._by_receiver):
        for _id, bucket in list(mapping.items()):
            if not bucket:
                mapping.pop(_id, None)