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
def _commit_version_unlocked(self, txn, version, origin):
    self._versions.append(version)
    self._prune_versions_unlocked()
    self.nodes = version.nodes
    if self.origin is None:
        self.origin = origin
    if txn is not None:
        self._end_write_unlocked(txn)