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
def _add_depends_on(self, revisions: Collection[Revision], map_: _RevisionMapType) -> None:
    """Resolve the 'dependencies' for each revision in a collection
        in terms of actual revision ids, as opposed to branch labels or other
        symbolic names.

        The collection is then assigned to the _resolved_dependencies
        attribute on each revision object.

        """
    for revision in revisions:
        if revision.dependencies:
            deps = [map_[dep] for dep in util.to_tuple(revision.dependencies)]
            revision._resolved_dependencies = tuple([d.revision for d in deps if d is not None])
        else:
            revision._resolved_dependencies = ()