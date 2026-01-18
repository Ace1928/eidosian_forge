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
def _append_param_insert_select_hasdefault(compiler: SQLCompiler, stmt: ValuesBase, c: ColumnClause[Any], values: List[_CrudParamElementSQLExpr], kw: Dict[str, Any]) -> None:
    if default_is_sequence(c.default):
        if compiler.dialect.supports_sequences and (not c.default.optional or not compiler.dialect.sequences_optional):
            values.append((c, compiler.preparer.format_column(c), c.default.next_value(), ()))
    elif default_is_clause_element(c.default):
        values.append((c, compiler.preparer.format_column(c), c.default.arg.self_group(), ()))
    else:
        values.append((c, compiler.preparer.format_column(c), _create_insert_prefetch_bind_param(compiler, c, process=False, **kw), (c.key,)))