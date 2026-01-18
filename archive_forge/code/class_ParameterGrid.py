import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..exceptions import NotFittedError
from ..metrics import check_scoring
from ..metrics._scorer import (
from ..utils import Bunch, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_method_params, check_is_fitted, indexable
from ._split import check_cv
from ._validation import (
class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    See Also
    --------
    GridSearchCV : Uses :class:`ParameterGrid` to perform a full parallelized
        parameter search.
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(f'Parameter grid should be a dict or a list, got: {param_grid!r} of type {type(param_grid).__name__}')
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f'Parameter grid is not a dict ({grid!r})')
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(f'Parameter array for {key!r} should be one-dimensional, got: {value!r} with shape {value.shape}')
                if isinstance(value, str) or not isinstance(value, (np.ndarray, Sequence)):
                    raise TypeError(f'Parameter grid for parameter {key!r} needs to be a list or a numpy array, but got {value!r} (of type {type(value).__name__}) instead. Single values need to be wrapped in a list with one element.')
                if len(value) == 0:
                    raise ValueError(f'Parameter grid for parameter {key!r} need to be a non-empty sequence, got: {value!r}')
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        product = partial(reduce, operator.mul)
        return sum((product((len(v) for v in p.values())) if p else 1 for p in self.param_grid))

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            The iteration index

        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        for sub_grid in self.param_grid:
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.prod(sizes)
            if ind >= total:
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out
        raise IndexError('ParameterGrid index out of range')