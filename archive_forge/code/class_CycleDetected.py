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
class CycleDetected(RevisionError):
    kind = 'Cycle'

    def __init__(self, revisions: Sequence[str]) -> None:
        self.revisions = revisions
        super().__init__('%s is detected in revisions (%s)' % (self.kind, ', '.join(revisions)))