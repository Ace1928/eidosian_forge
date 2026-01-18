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
class DDLIf(typing.NamedTuple):
    dialect: Optional[str]
    callable_: Optional[DDLIfCallable]
    state: Optional[Any]

    def _should_execute(self, ddl: BaseDDLElement, target: SchemaItem, bind: Optional[Connection], compiler: Optional[DDLCompiler]=None, **kw: Any) -> bool:
        if bind is not None:
            dialect = bind.dialect
        elif compiler is not None:
            dialect = compiler.dialect
        else:
            assert False, 'compiler or dialect is required'
        if isinstance(self.dialect, str):
            if self.dialect != dialect.name:
                return False
        elif isinstance(self.dialect, (tuple, list, set)):
            if dialect.name not in self.dialect:
                return False
        if self.callable_ is not None and (not self.callable_(ddl, target, bind, state=self.state, dialect=dialect, compiler=compiler, **kw)):
            return False
        return True