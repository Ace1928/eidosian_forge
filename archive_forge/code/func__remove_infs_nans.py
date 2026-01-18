import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def _remove_infs_nans(self, tensor: 'torch.Tensor') -> 'torch.Tensor':
    if not torch.isfinite(tensor).all():
        tensor = tensor[torch.isfinite(tensor)]
    return tensor