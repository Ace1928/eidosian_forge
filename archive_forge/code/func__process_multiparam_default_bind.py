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
def _process_multiparam_default_bind(compiler: SQLCompiler, stmt: ValuesBase, c: KeyedColumnElement[Any], index: int, kw: Dict[str, Any]) -> str:
    if not c.default:
        raise exc.CompileError('INSERT value for column %s is explicitly rendered as a boundparameter in the VALUES clause; a Python-side value or SQL expression is required' % c)
    elif default_is_clause_element(c.default):
        return compiler.process(c.default.arg.self_group(), **kw)
    elif c.default.is_sequence:
        return compiler.process(c.default, **kw)
    else:
        col = _multiparam_column(c, index)
        assert isinstance(stmt, dml.Insert)
        return _create_insert_prefetch_bind_param(compiler, col, process=True, **kw)