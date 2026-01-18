from itertools import product
from math import ceil, floor, sqrt
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, no_type_check
import numpy as np
import torch
from torch import Tensor
from torchmetrics.utilities.imports import _LATEX_AVAILABLE, _MATPLOTLIB_AVAILABLE, _SCIENCEPLOT_AVAILABLE
def _get_col_row_split(n: int) -> Tuple[int, int]:
    """Split `n` figures into `rows` x `cols` figures."""
    nsq = sqrt(n)
    if int(nsq) == nsq:
        return (int(nsq), int(nsq))
    if floor(nsq) * ceil(nsq) >= n:
        return (floor(nsq), ceil(nsq))
    return (ceil(nsq), ceil(nsq))