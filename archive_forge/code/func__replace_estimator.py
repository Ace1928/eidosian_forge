from abc import ABCMeta, abstractmethod
from contextlib import suppress
from typing import Any, List
import numpy as np
from ..base import BaseEstimator
from ..utils import _safe_indexing
from ..utils._tags import _safe_tags
from ._available_if import available_if
def _replace_estimator(self, attr, name, new_val):
    new_estimators = list(getattr(self, attr))
    for i, (estimator_name, _) in enumerate(new_estimators):
        if estimator_name == name:
            new_estimators[i] = (name, new_val)
            break
    setattr(self, attr, new_estimators)