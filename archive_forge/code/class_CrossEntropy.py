import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
@alias('ce')
class CrossEntropy(EvalMetric):
    """Computes Cross Entropy loss.

    The cross entropy over a batch of sample size :math:`N` is given by

    .. math::
       -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where :math:`t_{nk}=1` if and only if sample :math:`n` belongs to class :math:`k`.
    :math:`y_{nk}` denotes the probability of sample :math:`n` belonging to
    class :math:`k`.

    Parameters
    ----------
    eps : float
        Cross Entropy loss is undefined for predicted value is 0 or 1,
        so predicted values are added with the small constant.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> ce = mx.metric.CrossEntropy()
    >>> ce.update(labels, predicts)
    >>> print ce.get()
    ('cross-entropy', 0.57159948348999023)
    """

    def __init__(self, eps=1e-12, name='cross-entropy', output_names=None, label_names=None):
        super(CrossEntropy, self).__init__(name, eps=eps, output_names=output_names, label_names=label_names, has_global_stats=True)
        self.eps = eps

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()
            label = label.ravel()
            assert label.shape[0] == pred.shape[0]
            prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
            cross_entropy = (-numpy.log(prob + self.eps)).sum()
            self.sum_metric += cross_entropy
            self.global_sum_metric += cross_entropy
            self.num_inst += label.shape[0]
            self.global_num_inst += label.shape[0]