from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@classmethod
def _eval_condition_from_statement(cls, update_options, statement):
    mapper = update_options._subject_mapper
    target_cls = mapper.class_
    evaluator_compiler = evaluator._EvaluatorCompiler(target_cls)
    crit = ()
    if statement._where_criteria:
        crit += statement._where_criteria
    global_attributes = {}
    for opt in statement._with_options:
        if opt._is_criteria_option:
            opt.get_global_criteria(global_attributes)
    if global_attributes:
        crit += cls._adjust_for_extra_criteria(global_attributes, mapper)
    if crit:
        eval_condition = evaluator_compiler.process(*crit)
    else:

        def _eval_condition(obj):
            return True
        eval_condition = _eval_condition
    return eval_condition