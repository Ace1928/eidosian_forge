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
def _filter_into_branch_heads(self, targets: Iterable[Optional[_RevisionOrBase]]) -> Set[Optional[_RevisionOrBase]]:
    targets = set(targets)
    for rev in list(targets):
        assert rev
        if targets.intersection(self._get_descendant_nodes([rev], include_dependencies=False)).difference([rev]):
            targets.discard(rev)
    return targets