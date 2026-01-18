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
class DropSchema(_DropBase):
    """Represent a DROP SCHEMA statement.

    The argument here is the string name of the schema.

    """
    __visit_name__ = 'drop_schema'
    stringify_dialect = 'default'

    def __init__(self, name, cascade=False, if_exists=False):
        """Create a new :class:`.DropSchema` construct."""
        super().__init__(element=name, if_exists=if_exists)
        self.cascade = cascade