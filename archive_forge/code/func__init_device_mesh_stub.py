import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
def _init_device_mesh_stub():
    pass