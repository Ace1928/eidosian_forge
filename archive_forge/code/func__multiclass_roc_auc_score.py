import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import label_binarize
from ..utils import (
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import stable_cumsum
from ..utils.fixes import trapezoid
from ..utils.multiclass import type_of_target
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _check_sample_weight
from ._base import _average_binary_score, _average_multiclass_ovo_score
def _multiclass_roc_auc_score(y_true, y_score, labels, multi_class, average, sample_weight):
    """Multiclass roc auc score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class

    labels : array-like of shape (n_classes,) or None
        List of labels to index ``y_score`` used for multiclass. If ``None``,
        the lexical order of ``y_true`` is used to index ``y_score``.

    multi_class : {'ovr', 'ovo'}
        Determines the type of multiclass configuration to use.
        ``'ovr'``:
            Calculate metrics for the multiclass case using the one-vs-rest
            approach.
        ``'ovo'``:
            Calculate metrics for the multiclass case using the one-vs-one
            approach.

    average : {'micro', 'macro', 'weighted'}
        Determines the type of averaging performed on the pairwise binary
        metric scores
        ``'micro'``:
            Calculate metrics for the binarized-raveled classes. Only supported
            for `multi_class='ovr'`.

        .. versionadded:: 1.2

        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the
            prevalence of the classes.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights.

    """
    if not np.allclose(1, y_score.sum(axis=1)):
        raise ValueError('Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes')
    average_options = ('macro', 'weighted', None)
    if multi_class == 'ovr':
        average_options = ('micro',) + average_options
    if average not in average_options:
        raise ValueError('average must be one of {0} for multiclass problems'.format(average_options))
    multiclass_options = ('ovo', 'ovr')
    if multi_class not in multiclass_options:
        raise ValueError("multi_class='{0}' is not supported for multiclass ROC AUC, multi_class must be in {1}".format(multi_class, multiclass_options))
    if average is None and multi_class == 'ovo':
        raise NotImplementedError("average=None is not implemented for multi_class='ovo'.")
    if labels is not None:
        labels = column_or_1d(labels)
        classes = _unique(labels)
        if len(classes) != len(labels):
            raise ValueError("Parameter 'labels' must be unique")
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered")
        if len(classes) != y_score.shape[1]:
            raise ValueError("Number of given labels, {0}, not equal to the number of columns in 'y_score', {1}".format(len(classes), y_score.shape[1]))
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    else:
        classes = _unique(y_true)
        if len(classes) != y_score.shape[1]:
            raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
    if multi_class == 'ovo':
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported for multiclass one-vs-one ROC AUC, 'sample_weight' must be None in this case.")
        y_true_encoded = _encode(y_true, uniques=classes)
        return _average_multiclass_ovo_score(_binary_roc_auc_score, y_true_encoded, y_score, average=average)
    else:
        y_true_multilabel = label_binarize(y_true, classes=classes)
        return _average_binary_score(_binary_roc_auc_score, y_true_multilabel, y_score, average, sample_weight=sample_weight)