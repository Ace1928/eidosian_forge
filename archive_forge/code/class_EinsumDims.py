import itertools
from dataclasses import dataclass
from typing import List, Tuple
from torch.distributed._tensor.op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@dataclass
class EinsumDims:
    contracting_dims: List[str]
    batch_dims: List[str]
    lhs_out_only_dims: List[str]
    rhs_out_only_dims: List[str]

    @classmethod
    def parse_equation(cls, equation: str) -> Tuple[List[str], str]:
        """
        Parse the einsum equation str to input dim chars and output dim char
        """
        inputs, outputs = equation.split('->')
        input_dims, output_dims = (inputs.split(','), outputs.split(','))
        assert len(input_dims) <= 2, 'Only support at most two inputs'
        assert len(output_dims) == 1, 'Only support single output'
        output_dim = output_dims[0]
        return (input_dims, output_dim)

    @classmethod
    def parse_dims(cls, input_dims: List[str], output_dim: str) -> 'EinsumDims':
        """
        Parse the dims and extract the contracting, batch, and free dimensions
        for the left and right hand sides.
        """
        dim_char_set = set()
        for input_dim in input_dims:
            for input_char in list(input_dim):
                dim_char_set.add(input_char)
        all_dim_chars = sorted(dim_char_set)
        lhs_out_only_dims, rhs_out_only_dims = ([], [])
        batch_dims, contracting_dims = ([], [])
        for dim_char in all_dim_chars:
            if dim_char not in output_dim:
                contracting_dims.append(dim_char)
            else:
                is_batch_dim = True
                for input_dim in input_dims:
                    is_batch_dim = is_batch_dim and dim_char in input_dim
                if is_batch_dim:
                    batch_dims.append(dim_char)
                else:
                    assert len(input_dims) == 2, 'free dimension only supported for two inputs!'
                    lhs, rhs = input_dims
                    if dim_char in lhs:
                        lhs_out_only_dims.append(dim_char)
                    elif dim_char in rhs:
                        rhs_out_only_dims.append(dim_char)
                    else:
                        raise RuntimeError('Invalid dimension character')
        return cls(contracting_dims=contracting_dims, batch_dims=batch_dims, lhs_out_only_dims=lhs_out_only_dims, rhs_out_only_dims=rhs_out_only_dims)