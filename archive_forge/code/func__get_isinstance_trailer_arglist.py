from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi import settings
from jedi.inference.arguments import TreeArguments
from jedi.inference.value import iterable
from jedi.inference.base_value import NO_VALUES
from jedi.parser_utils import is_scope
def _get_isinstance_trailer_arglist(node):
    if node.type in ('power', 'atom_expr') and len(node.children) == 2:
        first, trailer = node.children
        if first.type == 'name' and first.value == 'isinstance' and (trailer.type == 'trailer') and (trailer.children[0] == '('):
            return trailer
    return None