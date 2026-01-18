import numpy as np
from scipy import sparse
from ._param_validation import StrOptions, validate_params
Estimate sample weights by class for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, list of dicts, "balanced", or None
        Weights associated with classes in the form `{class_label: weight}`.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        `[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]` instead of
        `[{1:1}, {2:5}, {3:1}, {4:1}]`.

        The `"balanced"` mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data:
        `n_samples / (n_classes * np.bincount(y))`.

        For multi-output, the weights of each column of y will be multiplied.

    y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)
        Array of original class labels per sample.

    indices : array-like of shape (n_subsample,), default=None
        Array of indices to be used in a subsample. Can be of length less than
        `n_samples` in the case of a subsample, or equal to `n_samples` in the
        case of a bootstrap subsample with repeated indices. If `None`, the
        sample weight will be calculated over the full sample. Only `"balanced"`
        is supported for `class_weight` if this is provided.

    Returns
    -------
    sample_weight_vect : ndarray of shape (n_samples,)
        Array with sample weights as applied to the original `y`.

    Examples
    --------
    >>> from sklearn.utils.class_weight import compute_sample_weight
    >>> y = [1, 1, 1, 1, 0, 0]
    >>> compute_sample_weight(class_weight="balanced", y=y)
    array([0.75, 0.75, 0.75, 0.75, 1.5 , 1.5 ])
    