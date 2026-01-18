from typing import Dict, Optional
from jedi.parser_utils import get_flow_branch_keyword, is_scope, get_parent_scope
from jedi.inference.recursion import execution_allowed
from jedi.inference.helpers import is_big_annoying_library
def _check_if(context, node):
    with execution_allowed(context.inference_state, node) as allowed:
        if not allowed:
            return UNSURE
        types = context.infer_node(node)
        values = set((x.py__bool__() for x in types))
        if len(values) == 1:
            return Status.lookup_table[values.pop()]
        else:
            return UNSURE