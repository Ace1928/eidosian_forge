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
def _infer_field(cls, field_name, is_instance):
    inference_state = cls.inference_state
    result = field_name.infer()
    for field_tree_instance in result:
        scalar_field = _infer_scalar_field(inference_state, field_name, field_tree_instance, is_instance)
        if scalar_field is not None:
            return scalar_field
        name = field_tree_instance.py__name__()
        is_many_to_many = name == 'ManyToManyField'
        if name in ('ForeignKey', 'OneToOneField') or is_many_to_many:
            if not is_instance:
                return _get_deferred_attributes(inference_state)
            values = _get_foreign_key_values(cls, field_tree_instance)
            if is_many_to_many:
                return ValueSet(filter(None, [_create_manager_for(v, 'RelatedManager') for v in values]))
            else:
                return values.execute_with_values()
    debug.dbg('django plugin: fail to infer `%s` from class `%s`', field_name.string_name, cls.py__name__())
    return result