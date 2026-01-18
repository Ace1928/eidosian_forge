import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
class Caffe(Loss):
    """Dummy metric for caffe criterions."""

    def __init__(self, name='caffe', output_names=None, label_names=None):
        super(Caffe, self).__init__(name, output_names=output_names, label_names=label_names)