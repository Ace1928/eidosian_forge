from jedi import settings
from jedi import debug
from jedi.parser_utils import get_parent_scope
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import TreeArguments
from jedi.inference.param import get_executed_param_names
from jedi.inference.helpers import is_stdlib_path
from jedi.inference.utils import to_list
from jedi.inference.value import instance
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.references import get_module_contexts_containing_name
from jedi.inference import recursion
def _get_potential_nodes(module_value, func_string_name):
    try:
        names = module_value.tree_node.get_used_names()[func_string_name]
    except KeyError:
        return
    for name in names:
        bracket = name.get_next_leaf()
        trailer = bracket.parent
        if trailer.type == 'trailer' and bracket == '(':
            yield (name, trailer)