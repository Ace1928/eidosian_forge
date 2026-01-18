from typing import Sequence, Tuple, TypeVar, Union
from ..model import Model
from ..types import ArrayXd, FloatsXd, IntsXd
Index into input arrays, and return the subarrays.

    This delegates to `array_getitem`, but allows type declarations.
    