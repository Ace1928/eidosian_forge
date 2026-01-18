from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def _whats_different(self, other: CacheKey) -> Iterator[str]:
    k1 = self.key
    k2 = other.key
    stack: List[int] = []
    pickup_index = 0
    while True:
        s1, s2 = (k1, k2)
        for idx in stack:
            s1 = s1[idx]
            s2 = s2[idx]
        for idx, (e1, e2) in enumerate(zip_longest(s1, s2)):
            if idx < pickup_index:
                continue
            if e1 != e2:
                if isinstance(e1, tuple) and isinstance(e2, tuple):
                    stack.append(idx)
                    break
                else:
                    yield ('key%s[%d]:  %s != %s' % (''.join(('[%d]' % id_ for id_ in stack)), idx, e1, e2))
        else:
            pickup_index = stack.pop(-1)
            break