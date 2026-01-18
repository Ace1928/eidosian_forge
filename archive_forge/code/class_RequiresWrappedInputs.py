from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
class RequiresWrappedInputs:
    """Used to mark, through inheritance,
    the fact that this class will require inputs to be passed as a single list"""
    pass