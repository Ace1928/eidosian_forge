from enum import Enum
from typing import List, Tuple, Union
from .tokenizers import (
from .implementations import (
class SplitDelimiterBehavior(Enum):
    REMOVED = 'removed'
    ISOLATED = 'isolated'
    MERGED_WITH_PREVIOUS = 'merged_with_previous'
    MERGED_WITH_NEXT = 'merged_with_next'
    CONTIGUOUS = 'contiguous'