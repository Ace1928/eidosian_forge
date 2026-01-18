import operator
import torch
from torch.export.exported_program import ConstantArgument, TensorArgument
from torch.fx.passes.infra.pass_base import PassBase, PassResult

    Performs constant folding and constant propagation.
    