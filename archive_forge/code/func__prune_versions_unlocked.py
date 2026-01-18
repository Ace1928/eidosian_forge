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
def _prune_versions_unlocked(self):
    assert len(self._versions) > 0
    if len(self._readers) > 0:
        least_kept = min((txn.version.id for txn in self._readers))
    else:
        least_kept = self._versions[-1].id
    while self._versions[0].id < least_kept and self._pruning_policy(self, self._versions[0]):
        self._versions.popleft()