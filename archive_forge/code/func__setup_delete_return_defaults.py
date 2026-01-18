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
def _setup_delete_return_defaults(compiler, stmt, compile_state, parameters, _getattr_col_key, _column_as_key, _col_bind_name, check_columns, values, toplevel, kw):
    _, _, implicit_return_defaults, *_ = _get_returning_modifiers(compiler, stmt, compile_state, toplevel)
    if not implicit_return_defaults:
        return
    if stmt._return_defaults_columns:
        compiler.implicit_returning.extend(implicit_return_defaults)
    if stmt._supplemental_returning:
        ir_set = set(compiler.implicit_returning)
        compiler.implicit_returning.extend((c for c in stmt._supplemental_returning if c not in ir_set))