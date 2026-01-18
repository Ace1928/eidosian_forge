import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from inspect import signature
from itertools import chain, combinations
from math import ceil, floor
import numpy as np
from scipy.special import comb
from ..utils import (
from ..utils._param_validation import Interval, RealNotInt, validate_params
from ..utils.metadata_routing import _MetadataRequester
from ..utils.multiclass import type_of_target
from ..utils.validation import _num_samples, check_array, column_or_1d
def _build_repr(self):
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    init_signature = signature(init)
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        warnings.simplefilter('always', FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, 'cvargs'):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == FutureWarning:
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value
    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))