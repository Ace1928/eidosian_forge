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
def _find_fields(cls):
    for name in _new_dict_filter(cls, is_instance=False).values():
        for value in name.infer():
            if value.name.get_qualified_names(include_module_names=True) == ('django', 'db', 'models', 'query_utils', 'DeferredAttribute'):
                yield name