from typing import Callable, Optional, TypeVar
from ..config import registry
from ..model import Model
from ..types import Floats2d
def _resize_parameter(name, layer, new_layer, filler=0):
    larger = new_layer.get_param(name)
    smaller = layer.get_param(name)
    larger[:len(smaller)] = smaller
    larger[len(smaller):] = filler
    new_layer.set_param(name, larger)