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
def _resolve_revision_number(self, id_: Optional[_GetRevArg]) -> Tuple[Tuple[str, ...], Optional[str]]:
    branch_label: Optional[str]
    if isinstance(id_, str) and '@' in id_:
        branch_label, id_ = id_.split('@', 1)
    elif id_ is not None and (isinstance(id_, tuple) and id_ and (not isinstance(id_[0], str)) or not isinstance(id_, (str, tuple))):
        raise RevisionError('revision identifier %r is not a string; ensure database driver settings are correct' % (id_,))
    else:
        branch_label = None
    self._revision_map
    if id_ == 'heads':
        if branch_label:
            return (self.filter_for_lineage(self.heads, branch_label), branch_label)
        else:
            return (self._real_heads, branch_label)
    elif id_ == 'head':
        current_head = self.get_current_head(branch_label)
        if current_head:
            return ((current_head,), branch_label)
        else:
            return ((), branch_label)
    elif id_ == 'base' or id_ is None:
        return ((), branch_label)
    else:
        return (util.to_tuple(id_, default=None), branch_label)