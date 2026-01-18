import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Placement, Replicate
from torch.distributed.device_mesh import DeviceMesh
Clear the existing sharding annotation first, by callling torch_xla.experimental.xla_sharding.clear_sharding API.