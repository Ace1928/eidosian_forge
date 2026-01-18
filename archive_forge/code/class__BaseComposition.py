from abc import ABCMeta, abstractmethod
from contextlib import suppress
from typing import Any, List
import numpy as np
from ..base import BaseEstimator
from ..utils import _safe_indexing
from ..utils._tags import _safe_tags
from ._available_if import available_if
class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for classifiers composed of named estimators."""
    steps: List[Any]

    @abstractmethod
    def __init__(self):
        pass

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        try:
            out.update(estimators)
        except (TypeError, ValueError):
            return out
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
        return out

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

    def _replace_estimator(self, attr, name, new_val):
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: {0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got {0!r}'.format(invalid_names))