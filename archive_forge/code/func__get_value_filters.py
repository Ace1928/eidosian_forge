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
def _get_value_filters(self, name_or_str):
    origin_scope = name_or_str if isinstance(name_or_str, Name) else None
    yield from self.get_filters(origin_scope=origin_scope)
    if self.is_stub():
        from jedi.inference.gradual.conversion import convert_values
        for c in convert_values(ValueSet({self})):
            yield from c.get_filters()