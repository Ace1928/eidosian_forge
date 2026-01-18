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
def _scan_insert_from_select_cols(compiler, stmt, compile_state, parameters, _getattr_col_key, _column_as_key, _col_bind_name, check_columns, values, toplevel, kw):
    cols = [stmt.table.c[_column_as_key(name)] for name in stmt._select_names]
    assert compiler.stack[-1]['selectable'] is stmt
    compiler.stack[-1]['insert_from_select'] = stmt.select
    add_select_cols: List[_CrudParamElementSQLExpr] = []
    if stmt.include_insert_from_select_defaults:
        col_set = set(cols)
        for col in stmt.table.columns:
            if col not in col_set and col.default and (not col.default.is_sentinel):
                cols.append(col)
    for c in cols:
        col_key = _getattr_col_key(c)
        if col_key in parameters and col_key not in check_columns:
            parameters.pop(col_key)
            values.append((c, compiler.preparer.format_column(c), None, ()))
        else:
            _append_param_insert_select_hasdefault(compiler, stmt, c, add_select_cols, kw)
    if add_select_cols:
        values.extend(add_select_cols)
        ins_from_select = compiler.stack[-1]['insert_from_select']
        if not isinstance(ins_from_select, Select):
            raise exc.CompileError(f"Can't extend statement for INSERT..FROM SELECT to include additional default-holding column(s) {', '.join((repr(key) for _, key, _, _ in add_select_cols))}.  Convert the selectable to a subquery() first, or pass include_defaults=False to Insert.from_select() to skip these columns.")
        ins_from_select = ins_from_select._generate()
        ins_from_select._raw_columns = list(ins_from_select._raw_columns) + [expr for _, _, expr, _ in add_select_cols]
        compiler.stack[-1]['insert_from_select'] = ins_from_select