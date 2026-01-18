import copy
from collections import deque
from typing import Iterable, List
from ray.data._internal.logical.interfaces import LogicalOperator, LogicalPlan, Rule
from ray.data._internal.logical.operators.one_to_one_operator import (
from ray.data._internal.logical.operators.read_operator import Read
def _apply_limit_fusion(self, op: LogicalOperator) -> LogicalOperator:
    """Given a DAG of LogicalOperators, traverse the DAG and fuse all
        back-to-back Limit operators, i.e.
        Limit[n] -> Limit[m] becomes Limit[min(n, m)].

        Returns a new LogicalOperator with the Limit operators fusion applied."""
    nodes: Iterable[LogicalOperator] = deque()
    for node in op.post_order_iter():
        nodes.appendleft(node)
    while len(nodes) > 0:
        current_op = nodes.pop()
        if isinstance(current_op, Limit):
            upstream_op = current_op.input_dependency
            if isinstance(upstream_op, Limit):
                new_limit = min(current_op._limit, upstream_op._limit)
                fused_limit_op = Limit(upstream_op.input_dependency, new_limit)
                fused_limit_op._input_dependencies = upstream_op.input_dependencies
                fused_limit_op._output_dependencies = current_op.output_dependencies
                upstream_input = upstream_op.input_dependency
                upstream_input._output_dependencies = [fused_limit_op]
                for current_output in current_op.output_dependencies:
                    current_output._input_dependencies = [fused_limit_op]
                nodes.append(fused_limit_op)
    return current_op