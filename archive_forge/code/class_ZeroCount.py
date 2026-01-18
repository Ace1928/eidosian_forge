import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
class ZeroCount(BaseEstimator, TransformerMixin):
    """Adds the count of zeros and count of non-zeros per sample as features."""

    def fit(self, X, y=None):
        """Dummy function to fit in with the sklearn API."""
        return self

    def transform(self, X, y=None):
        """Transform data by adding two virtual features.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components
            is the number of components.
        y: None
            Unused

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features)
            The transformed feature set
        """
        X = check_array(X)
        n_features = X.shape[1]
        X_transformed = np.copy(X)
        non_zero_vector = np.count_nonzero(X_transformed, axis=1)
        non_zero = np.reshape(non_zero_vector, (-1, 1))
        zero_col = np.reshape(n_features - non_zero_vector, (-1, 1))
        X_transformed = np.hstack((non_zero, X_transformed))
        X_transformed = np.hstack((zero_col, X_transformed))
        return X_transformed