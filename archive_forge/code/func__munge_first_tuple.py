from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import numpy.typing as npt
def _munge_first_tuple(tup: str) -> str:
    return 'dummy_' + tup.split(':', 1)[1]