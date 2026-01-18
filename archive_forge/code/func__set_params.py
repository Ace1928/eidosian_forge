from abc import ABCMeta, abstractmethod
from contextlib import suppress
from typing import Any, List
import numpy as np
from ..base import BaseEstimator
from ..utils import _safe_indexing
from ..utils._tags import _safe_tags
from ._available_if import available_if
def _set_params(self, attr, **params):
    if attr in params:
        setattr(self, attr, params.pop(attr))
    items = getattr(self, attr)
    if isinstance(items, list) and items:
        with suppress(TypeError):
            item_names, _ = zip(*items)
            for name in list(params.keys()):
                if '__' not in name and name in item_names:
                    self._replace_estimator(attr, name, params.pop(name))
    super().set_params(**params)
    return self