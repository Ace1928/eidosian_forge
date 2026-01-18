from __future__ import annotations
import collections
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Protocol
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import util as sqlautil
from .. import util
from ..util import not_none
def _get_ancestor_nodes(self, targets: Collection[Optional[_RevisionOrBase]], map_: Optional[_RevisionMapType]=None, check: bool=False, include_dependencies: bool=True) -> Iterator[Revision]:
    if include_dependencies:

        def fn(rev: Revision) -> Iterable[str]:
            return rev._normalized_down_revisions
    else:

        def fn(rev: Revision) -> Iterable[str]:
            return rev._versioned_down_revisions
    return self._iterate_related_revisions(fn, targets, map_=map_, check=check)