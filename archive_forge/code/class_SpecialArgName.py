from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
class SpecialArgName(Enum):
    possibly_redundant_memory_format = auto()