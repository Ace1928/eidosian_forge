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
def _check_read_only(self) -> None:
    if self.read_only:
        raise ReadOnly