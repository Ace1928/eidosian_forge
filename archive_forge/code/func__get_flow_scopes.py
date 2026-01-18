from typing import Dict, Optional
from jedi.parser_utils import get_flow_branch_keyword, is_scope, get_parent_scope
from jedi.inference.recursion import execution_allowed
from jedi.inference.helpers import is_big_annoying_library
def _get_flow_scopes(node):
    while True:
        node = get_parent_scope(node, include_flows=True)
        if node is None or is_scope(node):
            return
        yield node