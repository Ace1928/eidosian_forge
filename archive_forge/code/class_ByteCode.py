import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.jit.generate_bytecode import generate_upgraders_bytecode
from torchgen.code_template import CodeTemplate
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
class ByteCode(Enum):
    instructions = 1
    constants = 2
    types = 3
    operators = 4
    register_size = 5