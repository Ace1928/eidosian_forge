import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
class ExplanationPosition(Enum):
    BEFORE_DECISION = 0
    AFTER_DECISION = 1