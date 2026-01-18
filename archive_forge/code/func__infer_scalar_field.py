from inspect import Parameter
from jedi import debug
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.base_value import ValueSet, iterator_to_value_set, ValueWrapper
from jedi.inference.filters import DictFilter, AttributeOverwrite
from jedi.inference.names import NameWrapper, BaseTreeParamName
from jedi.inference.compiled.value import EmptyCompiledName
from jedi.inference.value.instance import TreeInstance
from jedi.inference.value.klass import ClassMixin
from jedi.inference.gradual.base import GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.signature import AbstractSignature
def _infer_scalar_field(inference_state, field_name, field_tree_instance, is_instance):
    try:
        module_name, attribute_name = mapping[field_tree_instance.py__name__()]
    except KeyError:
        return None
    if not is_instance:
        return _get_deferred_attributes(inference_state)
    if module_name is None:
        module = inference_state.builtins_module
    else:
        module = inference_state.import_module((module_name,))
    for attribute in module.py__getattribute__(attribute_name):
        return attribute.execute_with_values()