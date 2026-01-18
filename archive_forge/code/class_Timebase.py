import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
class Timebase:
    __annotations__ = {'numerator': int, 'denominator': int}
    __slots__ = ['numerator', 'denominator']

    def __init__(self, numerator: int, denominator: int) -> None:
        self.numerator = numerator
        self.denominator = denominator