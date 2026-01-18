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
def _apply_update_set_values_to_objects(cls, session, update_options, statement, matched_objects):
    """apply values to objects derived from an update statement, e.g.
        UPDATE..SET <values>

        """
    mapper = update_options._subject_mapper
    target_cls = mapper.class_
    evaluator_compiler = evaluator._EvaluatorCompiler(target_cls)
    resolved_values = cls._get_resolved_values(mapper, statement)
    resolved_keys_as_propnames = cls._resolved_keys_as_propnames(mapper, resolved_values)
    value_evaluators = {}
    for key, value in resolved_keys_as_propnames:
        try:
            _evaluator = evaluator_compiler.process(coercions.expect(roles.ExpressionElementRole, value))
        except evaluator.UnevaluatableError:
            pass
        else:
            value_evaluators[key] = _evaluator
    evaluated_keys = list(value_evaluators.keys())
    attrib = {k for k, v in resolved_keys_as_propnames}
    states = set()
    for obj, state, dict_ in matched_objects:
        to_evaluate = state.unmodified.intersection(evaluated_keys)
        for key in to_evaluate:
            if key in dict_:
                dict_[key] = value_evaluators[key](obj)
        state.manager.dispatch.refresh(state, None, to_evaluate)
        state._commit(dict_, list(to_evaluate))
        to_expire = attrib.intersection(dict_).difference(to_evaluate)
        if to_expire:
            state._expire_attributes(dict_, to_expire)
        states.add(state)
    session._register_altered(states)