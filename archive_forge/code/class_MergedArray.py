from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.helpers import get_int_or_none, is_string, \
from jedi.inference.utils import safe_property, to_list
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.filters import LazyAttributeOverwrite, publish_method
from jedi.inference.base_value import ValueSet, Value, NO_VALUES, \
from jedi.parser_utils import get_sync_comp_fors
from jedi.inference.context import CompForContext
from jedi.inference.value.dynamic_arrays import check_array_additions
class MergedArray(Sequence):

    def __init__(self, inference_state, arrays):
        super().__init__(inference_state)
        self.array_type = arrays[-1].array_type
        self._arrays = arrays

    def py__iter__(self, contextualized_node=None):
        for array in self._arrays:
            yield from array.py__iter__()

    def py__simple_getitem__(self, index):
        return ValueSet.from_sets((lazy_value.infer() for lazy_value in self.py__iter__()))