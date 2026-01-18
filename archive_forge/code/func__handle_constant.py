from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
def _handle_constant(self, hasconst):
    if hasconst is False or self.exog is None:
        self.k_constant = 0
        self.const_idx = None
    else:
        check_implicit = False
        exog_max = np.max(self.exog, axis=0)
        if not np.isfinite(exog_max).all():
            raise MissingDataError('exog contains inf or nans')
        exog_min = np.min(self.exog, axis=0)
        const_idx = np.where(exog_max == exog_min)[0].squeeze()
        self.k_constant = const_idx.size
        if self.k_constant == 1:
            if self.exog[:, const_idx].mean() != 0:
                self.const_idx = int(const_idx)
            else:
                check_implicit = True
        elif self.k_constant > 1:
            values = []
            for idx in const_idx:
                value = self.exog[:, idx].mean()
                if value == 1:
                    self.k_constant = 1
                    self.const_idx = int(idx)
                    break
                values.append(value)
            else:
                pos = np.array(values) != 0
                if pos.any():
                    self.k_constant = 1
                    self.const_idx = int(const_idx[pos.argmax()])
                else:
                    check_implicit = True
        elif self.k_constant == 0:
            check_implicit = True
        else:
            pass
        if check_implicit and (not hasconst):
            augmented_exog = np.column_stack((np.ones(self.exog.shape[0]), self.exog))
            rank_augm = np.linalg.matrix_rank(augmented_exog)
            rank_orig = np.linalg.matrix_rank(self.exog)
            self.k_constant = int(rank_orig == rank_augm)
            self.const_idx = None
        elif hasconst:
            self.k_constant = 1