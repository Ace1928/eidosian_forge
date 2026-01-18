from abc import abstractmethod
from typing import List
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.interfaces.optimizer import Rule
from ray.data._internal.logical.interfaces.physical_plan import PhysicalPlan
class EliminateBuildOutputBlocks(ZeroCopyMapFusionRule):
    """This rule eliminates unnecessary BuildOutputBlocksMapTransformFn,
    if the previous fn already outputs blocks.

    This happens for the "Read -> Map/Write" fusion.
    """

    def _optimize(self, transform_fns: List[MapTransformFn]) -> List[MapTransformFn]:
        new_transform_fns = []
        for i in range(len(transform_fns)):
            cur_fn = transform_fns[i]
            drop = False
            if i > 0 and i < len(transform_fns) - 1 and isinstance(cur_fn, BuildOutputBlocksMapTransformFn):
                prev_fn = transform_fns[i - 1]
                next_fn = transform_fns[i + 1]
                if prev_fn.output_type == MapTransformFnDataType.Block and next_fn.input_type == MapTransformFnDataType.Block:
                    drop = True
            if not drop:
                new_transform_fns.append(cur_fn)
        return new_transform_fns