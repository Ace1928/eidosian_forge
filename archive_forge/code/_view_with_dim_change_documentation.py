from typing import Tuple, Union, Sequence, cast
import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor import DTensor as DT
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import (

    infer the dtensor stride from a local tensor
    