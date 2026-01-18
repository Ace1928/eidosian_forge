import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
def _X_selected(X, selected):
    """Split X into selected features and other features"""
    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    non_sel = np.logical_not(sel)
    n_selected = np.sum(sel)
    X_sel = X[:, ind[sel]]
    X_not_sel = X[:, ind[non_sel]]
    return (X_sel, X_not_sel, n_selected, n_features)