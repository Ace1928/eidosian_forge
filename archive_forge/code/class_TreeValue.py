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
class TreeValue(Value):

    def __init__(self, inference_state, parent_context, tree_node):
        super().__init__(inference_state, parent_context)
        self.tree_node = tree_node

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.tree_node)