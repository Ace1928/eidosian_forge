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
def _get_all_current(self, id_: Tuple[str, ...]) -> Set[Optional[_RevisionOrBase]]:
    top_revs: Set[Optional[_RevisionOrBase]]
    top_revs = set(self.get_revisions(id_))
    top_revs.update(self._get_ancestor_nodes(list(top_revs), include_dependencies=True))
    return self._filter_into_branch_heads(top_revs)