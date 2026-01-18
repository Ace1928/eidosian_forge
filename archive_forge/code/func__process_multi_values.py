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
def _process_multi_values(self, statement: ValuesBase) -> None:
    for parameters in statement._multi_values:
        multi_parameters: List[MutableMapping[_DMLColumnElement, Any]] = [{c.key: value for c, value in zip(statement.table.c, parameter_set)} if isinstance(parameter_set, collections_abc.Sequence) else parameter_set for parameter_set in parameters]
        if self._no_parameters:
            self._no_parameters = False
            self._has_multi_parameters = True
            self._multi_parameters = multi_parameters
            self._dict_parameters = self._multi_parameters[0]
        elif not self._has_multi_parameters:
            self._cant_mix_formats_error()
        else:
            assert self._multi_parameters
            self._multi_parameters.extend(multi_parameters)