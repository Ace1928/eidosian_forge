import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
class _CategoricalBox(object):

    def __init__(self, data, contrast, levels):
        self.data = data
        self.contrast = contrast
        self.levels = levels
    __getstate__ = no_pickling