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
class _FakeSequence(Sequence):

    def __init__(self, inference_state, lazy_value_list):
        """
        type should be one of "tuple", "list"
        """
        super().__init__(inference_state)
        self._lazy_value_list = lazy_value_list

    def py__simple_getitem__(self, index):
        if isinstance(index, slice):
            return ValueSet([self])
        with reraise_getitem_errors(IndexError, TypeError):
            lazy_value = self._lazy_value_list[index]
        return lazy_value.infer()

    def py__iter__(self, contextualized_node=None):
        return self._lazy_value_list

    def py__bool__(self):
        return bool(len(self._lazy_value_list))

    def __repr__(self):
        return '<%s of %s>' % (type(self).__name__, self._lazy_value_list)