import re
from inspect import Parameter
from parso import ParserSyntaxError, parse
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.base import DefineGenericBaseClass, GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.gradual.type_var import TypeVar
from jedi.inference.helpers import is_string
from jedi.inference.compiled import builtin_from_name
from jedi.inference.param import get_executed_param_names
from jedi import debug
from jedi import parser_utils
def _infer_type_vars_for_callable(arguments, lazy_params):
    """
    Infers type vars for the Calllable class:

        def x() -> Callable[[Callable[..., _T]], _T]: ...
    """
    annotation_variable_results = {}
    for (_, lazy_value), lazy_callable_param in zip(arguments.unpack(), lazy_params):
        callable_param_values = lazy_callable_param.infer()
        actual_value_set = lazy_value.infer()
        merge_type_var_dicts(annotation_variable_results, callable_param_values.infer_type_vars(actual_value_set))
    return annotation_variable_results