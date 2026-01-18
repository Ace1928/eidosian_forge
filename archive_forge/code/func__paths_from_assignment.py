import os
import re
from pathlib import Path
from importlib.machinery import all_suffixes
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ContextualizedNode
from jedi.inference.helpers import is_string, get_str_or_none
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FileIO
from jedi import settings
from jedi import debug
def _paths_from_assignment(module_context, expr_stmt):
    """
    Extracts the assigned strings from an assignment that looks as follows::

        sys.path[0:0] = ['module/path', 'another/module/path']

    This function is in general pretty tolerant (and therefore 'buggy').
    However, it's not a big issue usually to add more paths to Jedi's sys_path,
    because it will only affect Jedi in very random situations and by adding
    more paths than necessary, it usually benefits the general user.
    """
    for assignee, operator in zip(expr_stmt.children[::2], expr_stmt.children[1::2]):
        try:
            assert operator in ['=', '+=']
            assert assignee.type in ('power', 'atom_expr') and len(assignee.children) > 1
            c = assignee.children
            assert c[0].type == 'name' and c[0].value == 'sys'
            trailer = c[1]
            assert trailer.children[0] == '.' and trailer.children[1].value == 'path'
            "\n            execution = c[2]\n            assert execution.children[0] == '['\n            subscript = execution.children[1]\n            assert subscript.type == 'subscript'\n            assert ':' in subscript.children\n            "
        except AssertionError:
            continue
        cn = ContextualizedNode(module_context.create_context(expr_stmt), expr_stmt)
        for lazy_value in cn.infer().iterate(cn):
            for value in lazy_value.infer():
                if is_string(value):
                    abs_path = _abs_path(module_context, value.get_safe_value())
                    if abs_path is not None:
                        yield abs_path