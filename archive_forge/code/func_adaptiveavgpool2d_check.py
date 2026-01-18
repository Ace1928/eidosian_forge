from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
def adaptiveavgpool2d_check(tensor_type, module_instance):
    output_size = module_instance.output_size
    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    elif isinstance(output_size, tuple):
        output_size = list(output_size)
        if output_size[0] is None:
            output_size[0] = output_size[1]
        if output_size[1] is None:
            output_size[1] = output_size[0]
    new_type_list = list(tensor_type.__args__)
    if len(tensor_type.__args__) == 4 or len(tensor_type.__args__) == 3:
        new_type_list[-1] = output_size[1]
        new_type_list[-2] = output_size[0]
        return TensorType(tuple(new_type_list))
    else:
        raise TypeError(f'Tensor ranks must be 3 or 4. Got {tensor_type}')