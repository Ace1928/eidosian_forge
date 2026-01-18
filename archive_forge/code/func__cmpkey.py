import collections
import itertools
import re
from typing import Any, Callable, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
def _cmpkey(epoch: int, release: Tuple[int, ...], pre: Optional[Tuple[str, int]], post: Optional[Tuple[str, int]], dev: Optional[Tuple[str, int]], local: Optional[Tuple[SubLocalType]]) -> CmpKey:
    _release = tuple(reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release)))))
    if pre is None and post is None and (dev is not None):
        _pre: PrePostDevType = NegativeInfinity
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre
    if post is None:
        _post: PrePostDevType = NegativeInfinity
    else:
        _post = post
    if dev is None:
        _dev: PrePostDevType = Infinity
    else:
        _dev = dev
    if local is None:
        _local: LocalType = NegativeInfinity
    else:
        _local = tuple(((i, '') if isinstance(i, int) else (NegativeInfinity, i) for i in local))
    return (epoch, _release, _pre, _post, _dev, _local)