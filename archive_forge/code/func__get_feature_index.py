def _get_feature_index(fx, feature_names=None):
    """Get feature index.

    Parameters
    ----------
    fx : int or str
        Feature index or name.

    feature_names : list of str, default=None
        All feature names from which to search the indices.

    Returns
    -------
    idx : int
        Feature index.
    """
    if isinstance(fx, str):
        if feature_names is None:
            raise ValueError(f'Cannot plot partial dependence for feature {fx!r} since the list of feature names was not provided, neither as column names of a pandas data-frame nor via the feature_names parameter.')
        try:
            return feature_names.index(fx)
        except ValueError as e:
            raise ValueError(f'Feature {fx!r} not in feature_names') from e
    return fx