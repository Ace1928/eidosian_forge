import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def _no_finite_values(self, tensor: 'torch.Tensor') -> bool:
    return tensor.shape == torch.Size([0]) or (~torch.isfinite(tensor)).all().item()