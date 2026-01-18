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
def _create_manager_for(cls, manager_cls='BaseManager'):
    managers = cls.inference_state.import_module(('django', 'db', 'models', 'manager')).py__getattribute__(manager_cls)
    for m in managers:
        if m.is_class_mixin():
            generics_manager = TupleGenericManager((ValueSet([cls]),))
            for c in GenericClass(m, generics_manager).execute_annotation():
                return c
    return None