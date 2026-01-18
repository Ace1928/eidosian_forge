import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def _generate_nd_grid(*sizes):
    coords = [torch.arange(s) for s in sizes]
    return torch.meshgrid(*coords)