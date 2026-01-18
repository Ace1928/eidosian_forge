import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def aatype_to_str_sequence(aatype: Sequence[int]) -> str:
    return ''.join([restypes_with_x[aatype[i]] for i in range(len(aatype))])