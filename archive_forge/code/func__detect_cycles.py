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
def _detect_cycles(self, rev_map: _InterimRevisionMapType, heads: Set[Revision], bases: Tuple[Revision, ...], _real_heads: Set[Revision], _real_bases: Tuple[Revision, ...]) -> None:
    if not rev_map:
        return
    if not heads or not bases:
        raise CycleDetected(list(rev_map))
    total_space = {rev.revision for rev in self._iterate_related_revisions(lambda r: r._versioned_down_revisions, heads, map_=cast(_RevisionMapType, rev_map))}.intersection((rev.revision for rev in self._iterate_related_revisions(lambda r: r.nextrev, bases, map_=cast(_RevisionMapType, rev_map))))
    deleted_revs = set(rev_map.keys()) - total_space
    if deleted_revs:
        raise CycleDetected(sorted(deleted_revs))
    if not _real_heads or not _real_bases:
        raise DependencyCycleDetected(list(rev_map))
    total_space = {rev.revision for rev in self._iterate_related_revisions(lambda r: r._all_down_revisions, _real_heads, map_=cast(_RevisionMapType, rev_map))}.intersection((rev.revision for rev in self._iterate_related_revisions(lambda r: r._all_nextrev, _real_bases, map_=cast(_RevisionMapType, rev_map))))
    deleted_revs = set(rev_map.keys()) - total_space
    if deleted_revs:
        raise DependencyCycleDetected(sorted(deleted_revs))