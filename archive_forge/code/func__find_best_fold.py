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
def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
    best_fold = None
    min_eval = np.inf
    min_samples_in_fold = np.inf
    for i in range(self.n_splits):
        y_counts_per_fold[i] += group_y_counts
        std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
        y_counts_per_fold[i] -= group_y_counts
        fold_eval = np.mean(std_per_class)
        samples_in_fold = np.sum(y_counts_per_fold[i])
        is_current_fold_better = fold_eval < min_eval or (np.isclose(fold_eval, min_eval) and samples_in_fold < min_samples_in_fold)
        if is_current_fold_better:
            min_eval = fold_eval
            min_samples_in_fold = samples_in_fold
            best_fold = i
    return best_fold