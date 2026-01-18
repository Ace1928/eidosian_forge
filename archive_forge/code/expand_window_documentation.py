from typing import Callable, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
For each vector in an input, construct an output vector that contains the
    input and a window of surrounding vectors. This is one step in a convolution.
    