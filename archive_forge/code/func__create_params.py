important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _create_params(parent, argslist_list):
    """
    `argslist_list` is a list that can contain an argslist as a first item, but
    most not. It's basically the items between the parameter brackets (which is
    at most one item).
    This function modifies the parser structure. It generates `Param` objects
    from the normal ast. Those param objects do not exist in a normal ast, but
    make the evaluation of the ast tree so much easier.
    You could also say that this function replaces the argslist node with a
    list of Param objects.
    """
    try:
        first = argslist_list[0]
    except IndexError:
        return []
    if first.type in ('name', 'fpdef'):
        return [Param([first], parent)]
    elif first == '*':
        return [first]
    else:
        if first.type == 'tfpdef':
            children = [first]
        else:
            children = first.children
        new_children = []
        start = 0
        for end, child in enumerate(children + [None], 1):
            if child is None or child == ',':
                param_children = children[start:end]
                if param_children:
                    if param_children[0] == '*' and (len(param_children) == 1 or param_children[1] == ',') or param_children[0] == '/':
                        for p in param_children:
                            p.parent = parent
                        new_children += param_children
                    else:
                        new_children.append(Param(param_children, parent))
                    start = end
        return new_children