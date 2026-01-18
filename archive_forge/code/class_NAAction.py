import statsmodels.tools.data as data_util
from patsy import dmatrices, NAAction
import numpy as np
class NAAction(NAAction):

    def _handle_NA_drop(self, values, is_NAs, origins):
        total_mask = np.zeros(is_NAs[0].shape[0], dtype=bool)
        for is_NA in is_NAs:
            total_mask |= is_NA
        good_mask = ~total_mask
        self.missing_mask = total_mask
        return [v[good_mask, ...] for v in values]