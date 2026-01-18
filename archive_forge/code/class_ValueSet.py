from functools import reduce
from operator import add
from itertools import zip_longest
from parso.python.tree import Name
from jedi import debug
from jedi.parser_utils import clean_scope_docstring
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.utils import safe_property
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method
class ValueSet:

    def __init__(self, iterable):
        self._set = frozenset(iterable)
        for value in iterable:
            assert not isinstance(value, ValueSet)

    @classmethod
    def _from_frozen_set(cls, frozenset_):
        self = cls.__new__(cls)
        self._set = frozenset_
        return self

    @classmethod
    def from_sets(cls, sets):
        """
        Used to work with an iterable of set.
        """
        aggregated = set()
        for set_ in sets:
            if isinstance(set_, ValueSet):
                aggregated |= set_._set
            else:
                aggregated |= frozenset(set_)
        return cls._from_frozen_set(frozenset(aggregated))

    def __or__(self, other):
        return self._from_frozen_set(self._set | other._set)

    def __and__(self, other):
        return self._from_frozen_set(self._set & other._set)

    def __iter__(self):
        return iter(self._set)

    def __bool__(self):
        return bool(self._set)

    def __len__(self):
        return len(self._set)

    def __repr__(self):
        return 'S{%s}' % ', '.join((str(s) for s in self._set))

    def filter(self, filter_func):
        return self.__class__(filter(filter_func, self._set))

    def __getattr__(self, name):

        def mapper(*args, **kwargs):
            return self.from_sets((getattr(value, name)(*args, **kwargs) for value in self._set))
        return mapper

    def __eq__(self, other):
        return self._set == other._set

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._set)

    def py__class__(self):
        return ValueSet((c.py__class__() for c in self._set))

    def iterate(self, contextualized_node=None, is_async=False):
        from jedi.inference.lazy_value import get_merged_lazy_value
        type_iters = [c.iterate(contextualized_node, is_async=is_async) for c in self._set]
        for lazy_values in zip_longest(*type_iters):
            yield get_merged_lazy_value([l for l in lazy_values if l is not None])

    def execute(self, arguments):
        return ValueSet.from_sets((c.inference_state.execute(c, arguments) for c in self._set))

    def execute_with_values(self, *args, **kwargs):
        return ValueSet.from_sets((c.execute_with_values(*args, **kwargs) for c in self._set))

    def goto(self, *args, **kwargs):
        return reduce(add, [c.goto(*args, **kwargs) for c in self._set], [])

    def py__getattribute__(self, *args, **kwargs):
        return ValueSet.from_sets((c.py__getattribute__(*args, **kwargs) for c in self._set))

    def get_item(self, *args, **kwargs):
        return ValueSet.from_sets((_getitem(c, *args, **kwargs) for c in self._set))

    def try_merge(self, function_name):
        value_set = self.__class__([])
        for c in self._set:
            try:
                method = getattr(c, function_name)
            except AttributeError:
                pass
            else:
                value_set |= method()
        return value_set

    def gather_annotation_classes(self):
        return ValueSet.from_sets([c.gather_annotation_classes() for c in self._set])

    def get_signatures(self):
        return [sig for c in self._set for sig in c.get_signatures()]

    def get_type_hint(self, add_class_info=True):
        t = [v.get_type_hint(add_class_info=add_class_info) for v in self._set]
        type_hints = sorted(filter(None, t))
        if len(type_hints) == 1:
            return type_hints[0]
        optional = 'None' in type_hints
        if optional:
            type_hints.remove('None')
        if len(type_hints) == 0:
            return None
        elif len(type_hints) == 1:
            s = type_hints[0]
        else:
            s = 'Union[%s]' % ', '.join(type_hints)
        if optional:
            s = 'Optional[%s]' % s
        return s

    def infer_type_vars(self, value_set):
        from jedi.inference.gradual.annotation import merge_type_var_dicts
        type_var_dict = {}
        for value in self._set:
            merge_type_var_dicts(type_var_dict, value.infer_type_vars(value_set))
        return type_var_dict