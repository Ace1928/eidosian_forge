import copy
import itertools
from parso.python import tree
from jedi import debug
from jedi import parser_utils
from jedi.inference.base_value import ValueSet, NO_VALUES, ContextualizedNode, \
from jedi.inference.lazy_value import LazyTreeValue
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import analysis
from jedi.inference import imports
from jedi.inference import arguments
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.value import iterable
from jedi.inference.value.dynamic_arrays import ListModification, DictModification
from jedi.inference.value import TreeInstance
from jedi.inference.helpers import is_string, is_literal, is_number, \
from jedi.inference.compiled.access import COMPARISON_OPERATORS
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.gradual.stub_value import VersionInfo
from jedi.inference.gradual import annotation
from jedi.inference.names import TreeNameDefinition
from jedi.inference.context import CompForContext
from jedi.inference.value.decorator import Decoratee
from jedi.plugins import plugin_manager
def _infer_comparison(context, left_values, operator, right_values):
    state = context.inference_state
    if isinstance(operator, str):
        operator_str = operator
    else:
        operator_str = str(operator.value)
    if not left_values or not right_values:
        result = (left_values or NO_VALUES) | (right_values or NO_VALUES)
        return _literals_to_types(state, result)
    elif operator_str == '|' and all((value.is_class() or value.is_compiled() for value in itertools.chain(left_values, right_values))):
        return ValueSet.from_sets((left_values, right_values))
    elif len(left_values) * len(right_values) > 6:
        return _literals_to_types(state, left_values | right_values)
    else:
        return ValueSet.from_sets((_infer_comparison_part(state, context, left, operator, right) for left in left_values for right in right_values))