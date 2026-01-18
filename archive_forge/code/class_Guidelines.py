import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
@dataclass
class Guidelines:
    categories: Sequence[Category]
    category_code_prefix: str = 'O'