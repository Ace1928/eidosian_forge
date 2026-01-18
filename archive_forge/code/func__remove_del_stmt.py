from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi import settings
from jedi.inference.arguments import TreeArguments
from jedi.inference.value import iterable
from jedi.inference.base_value import NO_VALUES
from jedi.parser_utils import is_scope
def _remove_del_stmt(names):
    for name in names:
        if name.tree_name is not None:
            definition = name.tree_name.get_definition()
            if definition is not None and definition.type == 'del_stmt':
                continue
        yield name