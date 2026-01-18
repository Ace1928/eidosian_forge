from jedi import debug
from jedi import settings
from jedi.inference import recursion
from jedi.inference.base_value import ValueSet, NO_VALUES, HelperValueMixin, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.cache import inference_state_method_cache
@inference_state_method_cache(default=NO_VALUES)
@debug.increase_indent
def _internal_check_array_additions(context, sequence):
    """
    Checks if a `Array` has "add" (append, insert, extend) statements:

    >>> a = [""]
    >>> a.append(1)
    """
    from jedi.inference import arguments
    debug.dbg('Dynamic array search for %s' % sequence, color='MAGENTA')
    module_context = context.get_root_context()
    if not settings.dynamic_array_additions or module_context.is_compiled():
        debug.dbg('Dynamic array search aborted.', color='MAGENTA')
        return NO_VALUES

    def find_additions(context, arglist, add_name):
        params = list(arguments.TreeArguments(context.inference_state, context, arglist).unpack())
        result = set()
        if add_name in ['insert']:
            params = params[1:]
        if add_name in ['append', 'add', 'insert']:
            for key, lazy_value in params:
                result.add(lazy_value)
        elif add_name in ['extend', 'update']:
            for key, lazy_value in params:
                result |= set(lazy_value.infer().iterate())
        return result
    temp_param_add, settings.dynamic_params_for_other_modules = (settings.dynamic_params_for_other_modules, False)
    is_list = sequence.name.string_name == 'list'
    search_names = ['append', 'extend', 'insert'] if is_list else ['add', 'update']
    added_types = set()
    for add_name in search_names:
        try:
            possible_names = module_context.tree_node.get_used_names()[add_name]
        except KeyError:
            continue
        else:
            for name in possible_names:
                value_node = context.tree_node
                if not value_node.start_pos < name.start_pos < value_node.end_pos:
                    continue
                trailer = name.parent
                power = trailer.parent
                trailer_pos = power.children.index(trailer)
                try:
                    execution_trailer = power.children[trailer_pos + 1]
                except IndexError:
                    continue
                else:
                    if execution_trailer.type != 'trailer' or execution_trailer.children[0] != '(' or execution_trailer.children[1] == ')':
                        continue
                random_context = context.create_context(name)
                with recursion.execution_allowed(context.inference_state, power) as allowed:
                    if allowed:
                        found = infer_call_of_leaf(random_context, name, cut_own_trailer=True)
                        if sequence in found:
                            added_types |= find_additions(random_context, execution_trailer.children[1], add_name)
    settings.dynamic_params_for_other_modules = temp_param_add
    debug.dbg('Dynamic array result %s', added_types, color='MAGENTA')
    return added_types