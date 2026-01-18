import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
class NNAPI_FuseCode:
    FUSED_NONE = 0
    FUSED_RELU = 1
    FUSED_RELU1 = 2
    FUSED_RELU6 = 3