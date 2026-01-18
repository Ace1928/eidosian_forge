from jedi.inference.compiled.value import CompiledValue, CompiledName, \
from jedi.inference.base_value import LazyValueWrapper
def _get_wrapped_value(self):
    instance, = builtin_from_name(self.inference_state, self._compiled_value.name.string_name).execute_with_values()
    return instance