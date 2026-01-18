from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
def _remap_type_vars(self, base):
    from jedi.inference.gradual.type_var import TypeVar
    filter = self._class_value.get_type_var_filter()
    for type_var_set in base.get_generics():
        new = NO_VALUES
        for type_var in type_var_set:
            if isinstance(type_var, TypeVar):
                names = filter.get(type_var.py__name__())
                new |= ValueSet.from_sets((name.infer() for name in names))
            else:
                new |= ValueSet([type_var])
        yield new