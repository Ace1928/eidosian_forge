import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
@alias('nll_loss')
class NegativeLogLikelihood(EvalMetric):
    """Computes the negative log-likelihood loss.

    The negative log-likelihoodd loss over a batch of sample size :math:`N` is given by

    .. math::
       -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where :math:`K` is the number of classes, :math:`y_{nk}` is the prediceted probability for
    :math:`k`-th class for :math:`n`-th sample. :math:`t_{nk}=1` if and only if sample
    :math:`n` belongs to class :math:`k`.

    Parameters
    ----------
    eps : float
        Negative log-likelihood loss is undefined for predicted value is 0,
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
    >>> nll_loss = mx.metric.NegativeLogLikelihood()
    >>> nll_loss.update(labels, predicts)
    >>> print nll_loss.get()
    ('nll-loss', 0.57159948348999023)
    """

    def __init__(self, eps=1e-12, name='nll-loss', output_names=None, label_names=None):
        super(NegativeLogLikelihood, self).__init__(name, eps=eps, output_names=output_names, label_names=label_names, has_global_stats=True)
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
            num_examples = pred.shape[0]
            assert label.shape[0] == num_examples, (label.shape[0], num_examples)
            prob = pred[numpy.arange(num_examples, dtype=numpy.int64), numpy.int64(label)]
            nll = (-numpy.log(prob + self.eps)).sum()
            self.sum_metric += nll
            self.global_sum_metric += nll
            self.num_inst += num_examples
            self.global_num_inst += num_examples