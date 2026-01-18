import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
class Torch(Loss):
    """Dummy metric for torch criterions."""

    def __init__(self, name='torch', output_names=None, label_names=None):
        super(Torch, self).__init__(name, output_names=output_names, label_names=label_names)