import collections
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.serial
import dns.ttl
def _checked_delete_name(self, name):
    for check in self._check_delete_name:
        check(self, name)
    self._delete_name(name)