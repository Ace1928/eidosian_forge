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
def _infer_comparison_part(inference_state, context, left, operator, right):
    l_is_num = is_number(left)
    r_is_num = is_number(right)
    if isinstance(operator, str):
        str_operator = operator
    else:
        str_operator = str(operator.value)
    if str_operator == '*':
        if isinstance(left, iterable.Sequence) or is_string(left):
            return ValueSet([left])
        elif isinstance(right, iterable.Sequence) or is_string(right):
            return ValueSet([right])
    elif str_operator == '+':
        if l_is_num and r_is_num or (is_string(left) and is_string(right)):
            return left.execute_operation(right, str_operator)
        elif _is_list(left) and _is_list(right) or (_is_tuple(left) and _is_tuple(right)):
            return ValueSet([iterable.MergedArray(inference_state, (left, right))])
    elif str_operator == '-':
        if l_is_num and r_is_num:
            return left.execute_operation(right, str_operator)
    elif str_operator == '%':
        return ValueSet([left])
    elif str_operator in COMPARISON_OPERATORS:
        if left.is_compiled() and right.is_compiled():
            result = left.execute_operation(right, str_operator)
            if result:
                return result
        else:
            if str_operator in ('is', '!=', '==', 'is not'):
                operation = COMPARISON_OPERATORS[str_operator]
                bool_ = operation(left, right)
                if (str_operator in ('is', '==')) == bool_:
                    return ValueSet([_bool_to_value(inference_state, bool_)])
            if isinstance(left, VersionInfo):
                version_info = _get_tuple_ints(right)
                if version_info is not None:
                    bool_result = compiled.access.COMPARISON_OPERATORS[operator](inference_state.environment.version_info, tuple(version_info))
                    return ValueSet([_bool_to_value(inference_state, bool_result)])
        return ValueSet([_bool_to_value(inference_state, True), _bool_to_value(inference_state, False)])
    elif str_operator in ('in', 'not in'):
        return NO_VALUES

    def check(obj):
        """Checks if a Jedi object is either a float or an int."""
        return isinstance(obj, TreeInstance) and obj.name.string_name in ('int', 'float')
    if str_operator in ('+', '-') and l_is_num != r_is_num and (not (check(left) or check(right))):
        message = 'TypeError: unsupported operand type(s) for +: %s and %s'
        analysis.add(context, 'type-error-operation', operator, message % (left, right))
    if left.is_class() or right.is_class():
        return NO_VALUES
    method_name = operator_to_magic_method[str_operator]
    magic_methods = left.py__getattribute__(method_name)
    if magic_methods:
        result = magic_methods.execute_with_values(right)
        if result:
            return result
    if not magic_methods:
        reverse_method_name = reverse_operator_to_magic_method[str_operator]
        magic_methods = right.py__getattribute__(reverse_method_name)
        result = magic_methods.execute_with_values(left)
        if result:
            return result
    result = ValueSet([left, right])
    debug.dbg('Used operator %s resulting in %s', operator, result)
    return result