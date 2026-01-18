from inspect import Parameter
from parso import tree
from jedi.inference.utils import to_list
from jedi.inference.names import ParamNameWrapper
from jedi.inference.helpers import is_big_annoying_library
def _iter_nodes_for_param(param_name):
    from parso.python.tree import search_ancestor
    from jedi.inference.arguments import TreeArguments
    execution_context = param_name.parent_context
    function_node = tree.search_ancestor(param_name.tree_name, 'funcdef', 'lambdef')
    module_node = function_node.get_root_node()
    start = function_node.children[-1].start_pos
    end = function_node.children[-1].end_pos
    for name in module_node.get_used_names().get(param_name.string_name):
        if start <= name.start_pos < end:
            argument = name.parent
            if argument.type == 'argument' and argument.children[0] == '*' * param_name.star_count:
                trailer = search_ancestor(argument, 'trailer')
                if trailer is not None:
                    context = execution_context.create_context(trailer)
                    if _goes_to_param_name(param_name, context, name):
                        values = _to_callables(context, trailer)
                        args = TreeArguments.create_cached(execution_context.inference_state, context=context, argument_node=trailer.children[1], trailer=trailer)
                        for c in values:
                            yield (c, args)