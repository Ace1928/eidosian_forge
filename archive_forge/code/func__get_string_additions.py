import os
from jedi.api import classes
from jedi.api.strings import StringName, get_quote_ending
from jedi.api.helpers import match
from jedi.inference.helpers import get_str_or_none
def _get_string_additions(module_context, start_leaf):

    def iterate_nodes():
        node = addition.parent
        was_addition = True
        for child_node in reversed(node.children[:node.children.index(addition)]):
            if was_addition:
                was_addition = False
                yield child_node
                continue
            if child_node != '+':
                break
            was_addition = True
    addition = start_leaf.get_previous_leaf()
    if addition != '+':
        return ''
    context = module_context.create_context(start_leaf)
    return _add_strings(context, reversed(list(iterate_nodes())))