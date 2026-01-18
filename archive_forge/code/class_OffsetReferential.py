from enum import Enum
from typing import List, Tuple, Union
from .tokenizers import (
from .implementations import (
class OffsetReferential(Enum):
    ORIGINAL = 'original'
    NORMALIZED = 'normalized'