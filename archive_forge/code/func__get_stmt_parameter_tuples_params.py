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
def _get_stmt_parameter_tuples_params(compiler, compile_state, parameters, stmt_parameter_tuples, _column_as_key, values, kw):
    for k, v in stmt_parameter_tuples:
        colkey = _column_as_key(k)
        if colkey is not None:
            parameters.setdefault(colkey, v)
        else:
            col_expr = compiler.process(k, include_table=compile_state.include_table_with_column_exprs)
            if coercions._is_literal(v):
                v = compiler.process(elements.BindParameter(None, v, type_=k.type), **kw)
            else:
                if v._is_bind_parameter and v.type._isnull:
                    v = v._with_binary_element_type(k.type)
                v = compiler.process(v.self_group(), **kw)
            values.append((k, col_expr, v, ()))