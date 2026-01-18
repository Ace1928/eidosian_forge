import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.decomposition import PCA
from .one_hot_encoder import OneHotEncoder, auto_select_categorical_features, _X_selected
Select continuous features and transform them using PCA.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        array-like, {n_samples, n_components}
        