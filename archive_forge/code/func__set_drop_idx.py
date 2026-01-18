import numbers
import warnings
from numbers import Integral
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..utils import _safe_indexing, check_array, is_scalar_nan
from ..utils._encode import _check_unknown, _encode, _get_counts, _unique
from ..utils._mask import _get_mask
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._set_output import _get_output_config
from ..utils.validation import _check_feature_names_in, check_is_fitted
def _set_drop_idx(self):
    """Compute the drop indices associated with `self.categories_`.

        If `self.drop` is:
        - `None`, No categories have been dropped.
        - `'first'`, All zeros to drop the first category.
        - `'if_binary'`, All zeros if the category is binary and `None`
          otherwise.
        - array-like, The indices of the categories that match the
          categories in `self.drop`. If the dropped category is an infrequent
          category, then the index for the infrequent category is used. This
          means that the entire infrequent category is dropped.

        This methods defines a public `drop_idx_` and a private
        `_drop_idx_after_grouping`.

        - `drop_idx_`: Public facing API that references the drop category in
          `self.categories_`.
        - `_drop_idx_after_grouping`: Used internally to drop categories *after* the
          infrequent categories are grouped together.

        If there are no infrequent categories or drop is `None`, then
        `drop_idx_=_drop_idx_after_grouping`.
        """
    if self.drop is None:
        drop_idx_after_grouping = None
    elif isinstance(self.drop, str):
        if self.drop == 'first':
            drop_idx_after_grouping = np.zeros(len(self.categories_), dtype=object)
        elif self.drop == 'if_binary':
            n_features_out_no_drop = [len(cat) for cat in self.categories_]
            if self._infrequent_enabled:
                for i, infreq_idx in enumerate(self._infrequent_indices):
                    if infreq_idx is None:
                        continue
                    n_features_out_no_drop[i] -= infreq_idx.size - 1
            drop_idx_after_grouping = np.array([0 if n_features_out == 2 else None for n_features_out in n_features_out_no_drop], dtype=object)
    else:
        drop_array = np.asarray(self.drop, dtype=object)
        droplen = len(drop_array)
        if droplen != len(self.categories_):
            msg = '`drop` should have length equal to the number of features ({}), got {}'
            raise ValueError(msg.format(len(self.categories_), droplen))
        missing_drops = []
        drop_indices = []
        for feature_idx, (drop_val, cat_list) in enumerate(zip(drop_array, self.categories_)):
            if not is_scalar_nan(drop_val):
                drop_idx = np.where(cat_list == drop_val)[0]
                if drop_idx.size:
                    drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0]))
                else:
                    missing_drops.append((feature_idx, drop_val))
                continue
            if is_scalar_nan(cat_list[-1]):
                drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, cat_list.size - 1))
            else:
                missing_drops.append((feature_idx, drop_val))
        if any(missing_drops):
            msg = 'The following categories were supposed to be dropped, but were not found in the training data.\n{}'.format('\n'.join(['Category: {}, Feature: {}'.format(c, v) for c, v in missing_drops]))
            raise ValueError(msg)
        drop_idx_after_grouping = np.array(drop_indices, dtype=object)
    self._drop_idx_after_grouping = drop_idx_after_grouping
    if not self._infrequent_enabled or drop_idx_after_grouping is None:
        self.drop_idx_ = self._drop_idx_after_grouping
    else:
        drop_idx_ = []
        for feature_idx, drop_idx in enumerate(drop_idx_after_grouping):
            default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
            if drop_idx is None or default_to_infrequent is None:
                orig_drop_idx = drop_idx
            else:
                orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]
            drop_idx_.append(orig_drop_idx)
        self.drop_idx_ = np.asarray(drop_idx_, dtype=object)