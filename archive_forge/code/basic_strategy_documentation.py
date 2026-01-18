import itertools
from dataclasses import dataclass
from typing import List, Tuple
from torch.distributed._tensor.op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh

        Parse the dims and extract the contracting, batch, and free dimensions
        for the left and right hand sides.
        