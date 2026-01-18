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
class SequenceLiteralValue(Sequence):
    _TUPLE_LIKE = ('testlist_star_expr', 'testlist', 'subscriptlist')
    mapping = {'(': 'tuple', '[': 'list', '{': 'set'}

    def __init__(self, inference_state, defining_context, atom):
        super().__init__(inference_state)
        self.atom = atom
        self._defining_context = defining_context
        if self.atom.type in self._TUPLE_LIKE:
            self.array_type = 'tuple'
        else:
            self.array_type = SequenceLiteralValue.mapping[atom.children[0]]
            'The builtin name of the array (list, set, tuple or dict).'

    def _get_generics(self):
        if self.array_type == 'tuple':
            return tuple((x.infer().py__class__() for x in self.py__iter__()))
        return super()._get_generics()

    def py__simple_getitem__(self, index):
        """Here the index is an int/str. Raises IndexError/KeyError."""
        if isinstance(index, slice):
            return ValueSet([self])
        else:
            with reraise_getitem_errors(TypeError, KeyError, IndexError):
                node = self.get_tree_entries()[index]
            if node == ':' or node.type == 'subscript':
                return NO_VALUES
            return self._defining_context.infer_node(node)

    def py__iter__(self, contextualized_node=None):
        """
        While values returns the possible values for any array field, this
        function returns the value for a certain index.
        """
        for node in self.get_tree_entries():
            if node == ':' or node.type == 'subscript':
                yield LazyKnownValue(Slice(self._defining_context, None, None, None))
            else:
                yield LazyTreeValue(self._defining_context, node)
        yield from check_array_additions(self._defining_context, self)

    def py__len__(self):
        return len(self.get_tree_entries())

    def get_tree_entries(self):
        c = self.atom.children
        if self.atom.type in self._TUPLE_LIKE:
            return c[::2]
        array_node = c[1]
        if array_node in (']', '}', ')'):
            return []
        if array_node.type == 'testlist_comp':
            return [value for value in array_node.children[::2] if value.type != 'star_expr']
        elif array_node.type == 'dictorsetmaker':
            kv = []
            iterator = iter(array_node.children)
            for key in iterator:
                if key == '**':
                    next(iterator)
                    next(iterator, None)
                else:
                    op = next(iterator, None)
                    if op is None or op == ',':
                        if key.type == 'star_expr':
                            pass
                        else:
                            kv.append(key)
                    else:
                        assert op == ':'
                        kv.append((key, next(iterator)))
                        next(iterator, None)
            return kv
        elif array_node.type == 'star_expr':
            return []
        else:
            return [array_node]

    def __repr__(self):
        return '<%s of %s>' % (self.__class__.__name__, self.atom)