from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
class AnalyzedFunction:
    __slots__ = ('analyzed_code', 'fn', 'closure_pywrappers', 'tracker_instrumented_fn', 'expr', 'bindparam_trackers', 'expected_expr', 'is_sequence', 'propagate_attrs', 'closure_bindparams')
    closure_bindparams: Optional[List[BindParameter[Any]]]
    expected_expr: Union[ClauseElement, List[ClauseElement]]
    bindparam_trackers: Optional[List[_BoundParameterGetter]]

    def __init__(self, analyzed_code, lambda_element, apply_propagate_attrs, fn):
        self.analyzed_code = analyzed_code
        self.fn = fn
        self.bindparam_trackers = analyzed_code.bindparam_trackers
        self._instrument_and_run_function(lambda_element)
        self._coerce_expression(lambda_element, apply_propagate_attrs)

    def _instrument_and_run_function(self, lambda_element):
        analyzed_code = self.analyzed_code
        fn = self.fn
        self.closure_pywrappers = closure_pywrappers = []
        build_py_wrappers = analyzed_code.build_py_wrappers
        if not build_py_wrappers:
            self.tracker_instrumented_fn = tracker_instrumented_fn = fn
            self.expr = lambda_element._invoke_user_fn(tracker_instrumented_fn)
        else:
            track_closure_variables = analyzed_code.track_closure_variables
            closure = fn.__closure__
            if closure:
                new_closure = {fv: cell.cell_contents for fv, cell in zip(fn.__code__.co_freevars, closure)}
            else:
                new_closure = {}
            new_globals = fn.__globals__.copy()
            for name, closure_index in build_py_wrappers:
                if closure_index is not None:
                    value = closure[closure_index].cell_contents
                    new_closure[name] = bind = PyWrapper(fn, name, value, closure_index=closure_index, track_bound_values=self.analyzed_code.track_bound_values)
                    if track_closure_variables:
                        closure_pywrappers.append(bind)
                else:
                    value = fn.__globals__[name]
                    new_globals[name] = bind = PyWrapper(fn, name, value)
            self.tracker_instrumented_fn = tracker_instrumented_fn = self._rewrite_code_obj(fn, [new_closure[name] for name in fn.__code__.co_freevars], new_globals)
            self.expr = lambda_element._invoke_user_fn(tracker_instrumented_fn)

    def _coerce_expression(self, lambda_element, apply_propagate_attrs):
        """Run the tracker-generated expression through coercion rules.

        After the user-defined lambda has been invoked to produce a statement
        for re-use, run it through coercion rules to both check that it's the
        correct type of object and also to coerce it to its useful form.

        """
        parent_lambda = lambda_element.parent_lambda
        expr = self.expr
        if parent_lambda is None:
            if isinstance(expr, collections_abc.Sequence):
                self.expected_expr = [cast('ClauseElement', coercions.expect(lambda_element.role, sub_expr, apply_propagate_attrs=apply_propagate_attrs)) for sub_expr in expr]
                self.is_sequence = True
            else:
                self.expected_expr = cast('ClauseElement', coercions.expect(lambda_element.role, expr, apply_propagate_attrs=apply_propagate_attrs))
                self.is_sequence = False
        else:
            self.expected_expr = expr
            self.is_sequence = False
        if apply_propagate_attrs is not None:
            self.propagate_attrs = apply_propagate_attrs._propagate_attrs
        else:
            self.propagate_attrs = util.EMPTY_DICT

    def _rewrite_code_obj(self, f, cell_values, globals_):
        """Return a copy of f, with a new closure and new globals

        yes it works in pypy :P

        """
        argrange = range(len(cell_values))
        code = 'def make_cells():\n'
        if cell_values:
            code += '    (%s) = (%s)\n' % (', '.join(('i%d' % i for i in argrange)), ', '.join(('o%d' % i for i in argrange)))
        code += '    def closure():\n'
        code += '        return %s\n' % ', '.join(('i%d' % i for i in argrange))
        code += '    return closure.__closure__'
        vars_ = {'o%d' % i: cell_values[i] for i in argrange}
        exec(code, vars_, vars_)
        closure = vars_['make_cells']()
        func = type(f)(f.__code__, globals_, f.__name__, f.__defaults__, closure)
        func.__annotations__ = f.__annotations__
        func.__kwdefaults__ = f.__kwdefaults__
        func.__doc__ = f.__doc__
        func.__module__ = f.__module__
        return func