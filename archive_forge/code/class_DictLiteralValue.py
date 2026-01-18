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
class DictLiteralValue(_DictMixin, SequenceLiteralValue, _DictKeyMixin):
    array_type = 'dict'

    def __init__(self, inference_state, defining_context, atom):
        Sequence.__init__(self, inference_state)
        self._defining_context = defining_context
        self.atom = atom

    def py__simple_getitem__(self, index):
        """Here the index is an int/str. Raises IndexError/KeyError."""
        compiled_value_index = compiled.create_simple_object(self.inference_state, index)
        for key, value in self.get_tree_entries():
            for k in self._defining_context.infer_node(key):
                for key_v in k.execute_operation(compiled_value_index, '=='):
                    if key_v.get_safe_value():
                        return self._defining_context.infer_node(value)
        raise SimpleGetItemNotFound('No key found in dictionary %s.' % self)

    def py__iter__(self, contextualized_node=None):
        """
        While values returns the possible values for any array field, this
        function returns the value for a certain index.
        """
        types = NO_VALUES
        for k, _ in self.get_tree_entries():
            types |= self._defining_context.infer_node(k)
        for _ in types:
            yield LazyKnownValues(types)

    @publish_method('values')
    def _imitate_values(self, arguments):
        lazy_value = LazyKnownValues(self._dict_values())
        return ValueSet([FakeList(self.inference_state, [lazy_value])])

    @publish_method('items')
    def _imitate_items(self, arguments):
        lazy_values = [LazyKnownValue(FakeTuple(self.inference_state, (LazyTreeValue(self._defining_context, key_node), LazyTreeValue(self._defining_context, value_node)))) for key_node, value_node in self.get_tree_entries()]
        return ValueSet([FakeList(self.inference_state, lazy_values)])

    def exact_key_items(self):
        """
        Returns a generator of tuples like dict.items(), where the key is
        resolved (as a string) and the values are still lazy values.
        """
        for key_node, value in self.get_tree_entries():
            for key in self._defining_context.infer_node(key_node):
                if is_string(key):
                    yield (key.get_safe_value(), LazyTreeValue(self._defining_context, value))

    def _dict_values(self):
        return ValueSet.from_sets((self._defining_context.infer_node(v) for k, v in self.get_tree_entries()))

    def _dict_keys(self):
        return ValueSet.from_sets((self._defining_context.infer_node(k) for k, v in self.get_tree_entries()))