from jedi import debug
from jedi import settings
from jedi.inference import recursion
from jedi.inference.base_value import ValueSet, NO_VALUES, HelperValueMixin, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.cache import inference_state_method_cache
class DictModification(_Modification):

    def py__iter__(self, contextualized_node=None):
        yield from self._wrapped_value.py__iter__(contextualized_node)
        yield self._contextualized_key

    def get_key_values(self):
        return self._wrapped_value.get_key_values() | self._contextualized_key.infer()