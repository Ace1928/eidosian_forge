import torch
import operator
import warnings
from typing import Callable, Dict, Iterable
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
from torch.fx.experimental.migrate_gradual_types.operation import \
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptive_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    avg_pool, counter = gen_tvar(counter)
    symbols[n] = avg_pool
    input_var = symbols[n.args[0]]
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)
    c2 = BinConstraintT(avg_pool, TensorType([d1, d2, module_instance.output_size[0], module_instance.output_size[1]]), op_eq)
    return ([c1, c2, *nat_constraints], counter)