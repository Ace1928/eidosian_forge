import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()
    USER_INPUT_MUTATION = auto()