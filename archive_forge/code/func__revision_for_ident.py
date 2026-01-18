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
def _revision_for_ident(self, resolved_id: Union[str, Tuple[()], None], check_branch: Optional[str]=None) -> Optional[Revision]:
    branch_rev: Optional[Revision]
    if check_branch:
        branch_rev = self._resolve_branch(check_branch)
    else:
        branch_rev = None
    revision: Union[Optional[Revision], Literal[False]]
    try:
        revision = self._revision_map[resolved_id]
    except KeyError:
        revision = False
    revs: Sequence[str]
    if revision is False:
        assert resolved_id
        revs = [x for x in self._revision_map if x and len(x) > 3 and x.startswith(resolved_id)]
        if branch_rev:
            revs = self.filter_for_lineage(revs, check_branch)
        if not revs:
            raise ResolutionError("No such revision or branch '%s'%s" % (resolved_id, '; please ensure at least four characters are present for partial revision identifier matches' if len(resolved_id) < 4 else ''), resolved_id)
        elif len(revs) > 1:
            raise ResolutionError("Multiple revisions start with '%s': %s..." % (resolved_id, ', '.join(("'%s'" % r for r in revs[0:3]))), resolved_id)
        else:
            revision = self._revision_map[revs[0]]
    if check_branch and revision is not None:
        assert branch_rev is not None
        assert resolved_id
        if not self._shares_lineage(revision.revision, branch_rev.revision):
            raise ResolutionError("Revision %s is not a member of branch '%s'" % (revision.revision, check_branch), resolved_id)
    return revision