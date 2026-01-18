import torch
from torch.distributed._tensor.op_schema import OpSchema, OpStrategy, OutputSharding
from torch.distributed._tensor.ops.basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.common_rules import einop_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def _mm_like_strategy(mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    self_strategy, mat2_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        assert strtg.input_specs is not None
        self_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        if is_tensor_shardable(self_strategy.output_shape, self_spec) and is_tensor_shardable(mat2_strategy.output_shape, mat2_spec):
            redistribute_cost = [generate_redistribute_costs(self_strategy, self_spec), generate_redistribute_costs(mat2_strategy, mat2_spec)]
            strtg.redistribute_cost = redistribute_cost
            filtered_strategies.append(strtg)
    mm_strategy.strategies = filtered_strategies
    return mm_strategy