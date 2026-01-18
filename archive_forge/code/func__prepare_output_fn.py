from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module
@staticmethod
def _prepare_output_fn(output_layouts, use_local_output, outputs, device_mesh):
    outputs = outputs.redistribute(placements=output_layouts)
    return outputs.to_local() if use_local_output else outputs