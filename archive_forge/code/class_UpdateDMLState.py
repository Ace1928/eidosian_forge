from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard
@CompileState.plugin_for('default', 'update')
class UpdateDMLState(DMLState):
    isupdate = True
    include_table_with_column_exprs = False

    def __init__(self, statement: Update, compiler: SQLCompiler, **kw: Any):
        self.statement = statement
        self.isupdate = True
        if statement._ordered_values is not None:
            self._process_ordered_values(statement)
        elif statement._values is not None:
            self._process_values(statement)
        elif statement._multi_values:
            self._no_multi_values_supported(statement)
        t, ef = self._make_extra_froms(statement)
        self._primary_table = t
        self._extra_froms = ef
        self.is_multitable = mt = ef
        self.include_table_with_column_exprs = bool(mt and compiler.render_table_with_column_in_update_from)

    def _process_ordered_values(self, statement: ValuesBase) -> None:
        parameters = statement._ordered_values
        if self._no_parameters:
            self._no_parameters = False
            assert parameters is not None
            self._dict_parameters = dict(parameters)
            self._ordered_values = parameters
            self._parameter_ordering = [key for key, value in parameters]
        else:
            raise exc.InvalidRequestError('Can only invoke ordered_values() once, and not mixed with any other values() call')