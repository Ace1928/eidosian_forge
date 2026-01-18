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
@util.memoized_property
def _revision_map(self) -> _RevisionMapType:
    """memoized attribute, initializes the revision map from the
        initial collection.

        """
    map_: _InterimRevisionMapType = sqlautil.OrderedDict()
    heads: Set[Revision] = sqlautil.OrderedSet()
    _real_heads: Set[Revision] = sqlautil.OrderedSet()
    bases: Tuple[Revision, ...] = ()
    _real_bases: Tuple[Revision, ...] = ()
    has_branch_labels = set()
    all_revisions = set()
    for revision in self._generator():
        all_revisions.add(revision)
        if revision.revision in map_:
            util.warn('Revision %s is present more than once' % revision.revision)
        map_[revision.revision] = revision
        if revision.branch_labels:
            has_branch_labels.add(revision)
        heads.add(revision)
        _real_heads.add(revision)
        if revision.is_base:
            bases += (revision,)
        if revision._is_real_base:
            _real_bases += (revision,)
    rev_map = map_.copy()
    self._map_branch_labels(has_branch_labels, cast(_RevisionMapType, map_))
    self._add_depends_on(all_revisions, cast(_RevisionMapType, map_))
    for rev in map_.values():
        for downrev in rev._all_down_revisions:
            if downrev not in map_:
                util.warn('Revision %s referenced from %s is not present' % (downrev, rev))
            down_revision = map_[downrev]
            down_revision.add_nextrev(rev)
            if downrev in rev._versioned_down_revisions:
                heads.discard(down_revision)
            _real_heads.discard(down_revision)
    self._normalize_depends_on(all_revisions, cast(_RevisionMapType, map_))
    self._detect_cycles(rev_map, heads, bases, _real_heads, _real_bases)
    revision_map: _RevisionMapType = dict(map_.items())
    revision_map[None] = revision_map[()] = None
    self.heads = tuple((rev.revision for rev in heads))
    self._real_heads = tuple((rev.revision for rev in _real_heads))
    self.bases = tuple((rev.revision for rev in bases))
    self._real_bases = tuple((rev.revision for rev in _real_bases))
    self._add_branches(has_branch_labels, revision_map)
    return revision_map