from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
@staticmethod
def _check_positive_width(width):
    if width <= 0.0:
        msg = 'Cannot add 0 or negative width smearing'
        raise ValueError(msg)