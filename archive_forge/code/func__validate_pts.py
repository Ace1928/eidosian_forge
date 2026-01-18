import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _validate_pts(pts_range: Tuple[int, int]) -> None:
    if pts_range[0] > pts_range[1] > 0:
        raise ValueError(f'Start pts should not be smaller than end pts, got start pts: {pts_range[0]} and end pts: {pts_range[1]}')