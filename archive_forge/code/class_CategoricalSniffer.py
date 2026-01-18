import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
class CategoricalSniffer(object):

    def __init__(self, NA_action, origin=None):
        self._NA_action = NA_action
        self._origin = origin
        self._contrast = None
        self._levels = None
        self._level_set = set()

    def levels_contrast(self):
        if self._levels is None:
            levels = list(self._level_set)
            levels.sort(key=SortAnythingKey)
            self._levels = levels
        return (tuple(self._levels), self._contrast)

    def sniff(self, data):
        if hasattr(data, 'contrast'):
            self._contrast = data.contrast
        if isinstance(data, _CategoricalBox):
            if data.levels is not None:
                self._levels = tuple(data.levels)
                return True
            else:
                data = data.data
        if safe_is_pandas_categorical(data):
            self._levels = tuple(pandas_Categorical_categories(data))
            return True
        if hasattr(data, 'dtype') and safe_issubdtype(data.dtype, np.bool_):
            self._level_set = set([True, False])
            return True
        data = _categorical_shape_fix(data)
        for value in data:
            if self._NA_action.is_categorical_NA(value):
                continue
            if value is True or value is False:
                self._level_set.update([True, False])
            else:
                try:
                    self._level_set.add(value)
                except TypeError:
                    raise PatsyError('Error interpreting categorical data: all items must be hashable', self._origin)
        return self._level_set == set([True, False])
    __getstate__ = no_pickling