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
@safe_property
@memoize_method
def _wrapped_value(self):
    with debug.increase_indent_cm('Resolve lazy value wrapper'):
        return self._get_wrapped_value()