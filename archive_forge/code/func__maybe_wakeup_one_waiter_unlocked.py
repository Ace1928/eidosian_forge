import collections
import threading
from typing import Callable, Deque, Optional, Set, Union
import dns.exception
import dns.immutable
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.zone
def _maybe_wakeup_one_waiter_unlocked(self):
    if len(self._write_waiters) > 0:
        self._write_event = self._write_waiters.popleft()
        self._write_event.set()