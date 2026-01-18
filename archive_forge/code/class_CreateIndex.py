from __future__ import annotations
import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self
class CreateIndex(_CreateBase):
    """Represent a CREATE INDEX statement."""
    __visit_name__ = 'create_index'

    def __init__(self, element, if_not_exists=False):
        """Create a :class:`.Createindex` construct.

        :param element: a :class:`_schema.Index` that's the subject
         of the CREATE.
        :param if_not_exists: if True, an IF NOT EXISTS operator will be
         applied to the construct.

         .. versionadded:: 1.4.0b2

        """
        super().__init__(element, if_not_exists=if_not_exists)