import copy
import itertools
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT, MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj, Constraint, DVar, TVar, \
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F, GetItem, GetItemTensor, IndexSelect
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar
from torch.fx.tensor_type import TensorType, Dyn
from typing import Callable, Dict, List

    Simulates broadcasting on e1 and e2 and returns the results
    respectively in e11 and e12. Because of gradual types,
    e1 and e2 may not be equal. Similarly, e11 and e12 may not
    be equal. e11 and e12 should be guaranteed to be consistent
    as they represent the shapes of the tensors to be added after
    broadcasting.
    Args:
        e1: TVar representing the type of input 1
        e2: TVar representing the type of input 2
        e11: TVar representing the representing broadcasted input 1
        e12: TVar representing the representing broadcasted input 2
        i: The rank of the resulting type of addition
        counter: for variable tracking

    Returns: Simplified broadcasting constraints

    