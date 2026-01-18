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
class InvokeCreateDDLBase(InvokeDDLBase):

    @contextlib.contextmanager
    def with_ddl_events(self, target, **kw):
        """helper context manager that will apply appropriate DDL events
        to a CREATE or DROP operation."""
        target.dispatch.before_create(target, self.connection, _ddl_runner=self, **kw)
        yield
        target.dispatch.after_create(target, self.connection, _ddl_runner=self, **kw)