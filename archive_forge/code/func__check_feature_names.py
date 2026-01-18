def _check_feature_names(X, feature_names=None):
    """Check feature names.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    feature_names : None or array-like of shape (n_names,), dtype=str
        Feature names to check or `None`.

    Returns
    -------
    feature_names : list of str
        Feature names validated. If `feature_names` is `None`, then a list of
        feature names is provided, i.e. the column names of a pandas dataframe
        or a generic list of feature names (e.g. `["x0", "x1", ...]`) for a
        NumPy array.
    """
    if feature_names is None:
        if hasattr(X, 'columns') and hasattr(X.columns, 'tolist'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'x{i}' for i in range(X.shape[1])]
    elif hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('feature_names should not contain duplicates.')
    return feature_names