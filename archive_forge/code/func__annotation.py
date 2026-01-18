from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
@property
def _annotation(self):
    if self.value.is_class():
        return None
    return self._function_value.tree_node.annotation