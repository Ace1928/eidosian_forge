from numbers import Integral
import numpy as np
from ..base import _fit_context
from ..metrics import pairwise_distances_chunked
from ..metrics.pairwise import _NAN_METRICS
from ..neighbors._base import _get_weights
from ..utils import is_scalar_nan
from ..utils._mask import _get_mask
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
from ._base import _BaseImputer
def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
    """Helper function to impute a single column.

        Parameters
        ----------
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.

        n_neighbors : int
            Number of neighbors to consider.

        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.

        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.

        Returns
        -------
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
    donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[:, :n_neighbors]
    donors_dist = dist_pot_donors[np.arange(donors_idx.shape[0])[:, None], donors_idx]
    weight_matrix = _get_weights(donors_dist, self.weights)
    if weight_matrix is not None:
        weight_matrix[np.isnan(weight_matrix)] = 0.0
    donors = fit_X_col.take(donors_idx)
    donors_mask = mask_fit_X_col.take(donors_idx)
    donors = np.ma.array(donors, mask=donors_mask)
    return np.ma.average(donors, axis=1, weights=weight_matrix).data