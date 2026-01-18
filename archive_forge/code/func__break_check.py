from typing import Dict, Optional
from jedi.parser_utils import get_flow_branch_keyword, is_scope, get_parent_scope
from jedi.inference.recursion import execution_allowed
from jedi.inference.helpers import is_big_annoying_library
def _break_check(context, value_scope, flow_scope, node):
    reachable = REACHABLE
    if flow_scope.type == 'if_stmt':
        if flow_scope.is_node_after_else(node):
            for check_node in flow_scope.get_test_nodes():
                reachable = _check_if(context, check_node)
                if reachable in (REACHABLE, UNSURE):
                    break
            reachable = reachable.invert()
        else:
            flow_node = flow_scope.get_corresponding_test_node(node)
            if flow_node is not None:
                reachable = _check_if(context, flow_node)
    elif flow_scope.type in ('try_stmt', 'while_stmt'):
        return UNSURE
    if reachable in (UNREACHABLE, UNSURE):
        return reachable
    if value_scope != flow_scope and value_scope != flow_scope.parent:
        flow_scope = get_parent_scope(flow_scope, include_flows=True)
        return reachable & _break_check(context, value_scope, flow_scope, node)
    else:
        return reachable