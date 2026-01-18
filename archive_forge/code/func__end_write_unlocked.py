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
def _end_write_unlocked(self, txn):
    assert self._write_txn == txn
    self._write_txn = None
    self._maybe_wakeup_one_waiter_unlocked()