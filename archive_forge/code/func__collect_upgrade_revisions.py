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
def _collect_upgrade_revisions(self, upper: _RevisionIdentifierType, lower: _RevisionIdentifierType, inclusive: bool, implicit_base: bool, assert_relative_length: bool) -> Tuple[Set[Revision], Tuple[Revision, ...]]:
    """
        Compute the set of required revisions specified by :upper, and the
        current set of active revisions specified by :lower. Find the
        difference between the two to compute the required upgrades.

        :inclusive=True includes the current/lower revisions in the set

        :implicit_base=False only returns revisions which are downstream
        of the current/lower revisions. Dependencies from branches with
        different bases will not be included.
        """
    targets: Collection[Revision] = [is_revision(rev) for rev in self._parse_upgrade_target(current_revisions=lower, target=upper, assert_relative_length=assert_relative_length)]
    if isinstance(lower, str) and '@' in lower:
        branch, _, _ = lower.partition('@')
        branch_rev = self.get_revision(branch)
        if branch_rev is not None and branch_rev.revision == branch:
            assert len(branch_rev.branch_labels) == 1
            branch = next(iter(branch_rev.branch_labels))
        targets = {need for need in targets if branch in need.branch_labels}
    required_node_set = set(self._get_ancestor_nodes(targets, check=True, include_dependencies=True)).union(targets)
    current_revisions = self.get_revisions(lower)
    if not implicit_base and any((rev not in required_node_set for rev in current_revisions if rev is not None)):
        raise RangeNotAncestorError(lower, upper)
    assert type(current_revisions) is tuple, 'current_revisions should be a tuple'
    if current_revisions and current_revisions[0] is None:
        _, rev = self._parse_downgrade_target(current_revisions=upper, target=lower, assert_relative_length=assert_relative_length)
        assert rev
        if rev == 'base':
            current_revisions = tuple()
            lower = None
        else:
            current_revisions = (rev,)
            lower = rev.revision
    current_node_set = set(self._get_ancestor_nodes(current_revisions, check=True, include_dependencies=True)).union(current_revisions)
    needs = required_node_set.difference(current_node_set)
    if inclusive:
        needs.update((is_revision(rev) for rev in self.get_revisions(lower)))
    if current_revisions and (not implicit_base):
        lower_descendents = self._get_descendant_nodes([is_revision(rev) for rev in current_revisions], check=True, include_dependencies=False)
        needs.intersection_update(lower_descendents)
    return (needs, tuple(targets))