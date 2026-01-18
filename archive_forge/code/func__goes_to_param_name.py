from inspect import Parameter
from parso import tree
from jedi.inference.utils import to_list
from jedi.inference.names import ParamNameWrapper
from jedi.inference.helpers import is_big_annoying_library
def _goes_to_param_name(param_name, context, potential_name):
    if potential_name.type != 'name':
        return False
    from jedi.inference.names import TreeNameDefinition
    found = TreeNameDefinition(context, potential_name).goto()
    return any((param_name.parent_context == p.parent_context and param_name.start_pos == p.start_pos for p in found))