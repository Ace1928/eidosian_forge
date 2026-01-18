from __future__ import annotations
import functools
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import coercions
from . import dml
from . import elements
from . import roles
from .base import _DefaultDescriptionTuple
from .dml import isinsert as _compile_state_isinsert
from .elements import ColumnClause
from .schema import default_is_clause_element
from .schema import default_is_sequence
from .selectable import Select
from .selectable import TableClause
from .. import exc
from .. import util
from ..util.typing import Literal
def _key_getters_for_crud_column(compiler: SQLCompiler, stmt: ValuesBase, compile_state: DMLState) -> Tuple[Callable[[Union[str, ColumnClause[Any]]], Union[str, Tuple[str, str]]], Callable[[ColumnClause[Any]], Union[str, Tuple[str, str]]], _BindNameForColProtocol]:
    if dml.isupdate(compile_state) and compile_state._extra_froms:
        _et = set(compile_state._extra_froms)
        c_key_role = functools.partial(coercions.expect_as_key, roles.DMLColumnRole)

        def _column_as_key(key: Union[ColumnClause[Any], str]) -> Union[str, Tuple[str, str]]:
            str_key = c_key_role(key)
            if hasattr(key, 'table') and key.table in _et:
                return (key.table.name, str_key)
            else:
                return str_key

        def _getattr_col_key(col: ColumnClause[Any]) -> Union[str, Tuple[str, str]]:
            if col.table in _et:
                return (col.table.name, col.key)
            else:
                return col.key

        def _col_bind_name(col: ColumnClause[Any]) -> str:
            if col.table in _et:
                if TYPE_CHECKING:
                    assert isinstance(col.table, TableClause)
                return '%s_%s' % (col.table.name, col.key)
            else:
                return col.key
    else:
        _column_as_key = functools.partial(coercions.expect_as_key, roles.DMLColumnRole)
        _getattr_col_key = _col_bind_name = operator.attrgetter('key')
    return (_column_as_key, _getattr_col_key, _col_bind_name)