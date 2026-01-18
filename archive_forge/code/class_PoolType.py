import math
from dataclasses import dataclass
from enum import Enum
import torch
class PoolType(str, Enum):
    Conv2D = 'CONV_2D'