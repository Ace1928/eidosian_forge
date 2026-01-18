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
def _parse_upgrade_target(self, current_revisions: _RevisionIdentifierType, target: _RevisionIdentifierType, assert_relative_length: bool) -> Tuple[Optional[_RevisionOrBase], ...]:
    """
        Parse upgrade command syntax :target to retrieve the target revision
        and given the :current_revisions stamp of the database.

        Returns a tuple of Revision objects which should be iterated/upgraded
        to. The target may be specified in absolute form, or relative to
        :current_revisions.
        """
    if isinstance(target, str):
        match = _relative_destination.match(target)
    else:
        match = None
    if not match:
        return self.get_revisions(target)
    current_revisions_tup: Union[str, Tuple[Optional[str], ...], None]
    current_revisions_tup = util.to_tuple(current_revisions)
    branch_label, symbol, relative_str = match.groups()
    relative = int(relative_str)
    if relative > 0:
        if symbol is None:
            if not current_revisions_tup:
                current_revisions_tup = (None,)
            start_revs = current_revisions_tup
            if branch_label:
                start_revs = self.filter_for_lineage(self.get_revisions(current_revisions_tup), branch_label)
                if not start_revs:
                    active_on_branch = self.filter_for_lineage(self._get_ancestor_nodes(self.get_revisions(current_revisions_tup)), branch_label)
                    start_revs = tuple({rev.revision for rev in active_on_branch} - {down for rev in active_on_branch for down in rev._normalized_down_revisions})
                    if not start_revs:
                        start_revs = (None,)
            if len(start_revs) > 1:
                raise RevisionError('Ambiguous upgrade from multiple current revisions')
            rev = self._walk(start=start_revs[0], steps=relative, branch_label=branch_label, no_overwalk=assert_relative_length)
            if rev is None:
                raise RevisionError("Relative revision %s didn't produce %d migrations" % (relative_str, abs(relative)))
            return (rev,)
        else:
            return (self._walk(start=self.get_revision(symbol), steps=relative, branch_label=branch_label, no_overwalk=assert_relative_length),)
    else:
        if symbol is None:
            raise RevisionError("Relative revision %s didn't produce %d migrations" % (relative, abs(relative)))
        return (self._walk(start=self.get_revision(symbol) if branch_label is None else self.get_revision('%s@%s' % (branch_label, symbol)), steps=relative, no_overwalk=assert_relative_length),)