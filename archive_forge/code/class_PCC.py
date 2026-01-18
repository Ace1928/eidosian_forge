import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
class PCC(EvalMetric):
    """PCC is a multiclass equivalent for the Matthews correlation coefficient derived
    from a discrete solution to the Pearson correlation coefficient.

    .. math::
        \\text{PCC} = \\frac {\\sum _{k}\\sum _{l}\\sum _{m}C_{kk}C_{lm}-C_{kl}C_{mk}}
        {{\\sqrt {\\sum _{k}(\\sum _{l}C_{kl})(\\sum _{k'|k'\\neq k}\\sum _{l'}C_{k'l'})}}
         {\\sqrt {\\sum _{k}(\\sum _{l}C_{lk})(\\sum _{k'|k'\\neq k}\\sum _{l'}C_{l'k'})}}}

    defined in terms of a K x K confusion matrix C.

    When there are more than two labels the PCC will no longer range between -1 and +1.
    Instead the minimum value will be between -1 and 0 depending on the true distribution.
    The maximum value is always +1.

    Parameters
    ----------
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
    >>> # In this example the network almost always predicts positive
    >>> false_positives = 1000
    >>> false_negatives = 1
    >>> true_positives = 10000
    >>> true_negatives = 1
    >>> predicts = [mx.nd.array(
        [[.3, .7]]*false_positives +
        [[.7, .3]]*true_negatives +
        [[.7, .3]]*false_negatives +
        [[.3, .7]]*true_positives
    )]
    >>> labels  = [mx.nd.array(
        [0]*(false_positives + true_negatives) +
        [1]*(false_negatives + true_positives)
    )]
    >>> f1 = mx.metric.F1()
    >>> f1.update(preds = predicts, labels = labels)
    >>> pcc = mx.metric.PCC()
    >>> pcc.update(preds = predicts, labels = labels)
    >>> print f1.get()
    ('f1', 0.95233560306652054)
    >>> print pcc.get()
    ('pcc', 0.01917751877733392)
    """

    def __init__(self, name='pcc', output_names=None, label_names=None, has_global_stats=True):
        self.k = 2
        super(PCC, self).__init__(name=name, output_names=output_names, label_names=label_names, has_global_stats=has_global_stats)

    def _grow(self, inc):
        self.lcm = numpy.pad(self.lcm, ((0, inc), (0, inc)), 'constant', constant_values=0)
        self.gcm = numpy.pad(self.gcm, ((0, inc), (0, inc)), 'constant', constant_values=0)
        self.k += inc

    def _calc_mcc(self, cmat):
        n = cmat.sum()
        x = cmat.sum(axis=1)
        y = cmat.sum(axis=0)
        cov_xx = numpy.sum(x * (n - x))
        cov_yy = numpy.sum(y * (n - y))
        if cov_xx == 0 or cov_yy == 0:
            return float('nan')
        i = cmat.diagonal()
        cov_xy = numpy.sum(i * n - x * y)
        return cov_xy / (cov_xx * cov_yy) ** 0.5

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
            label = label.astype('int32', copy=False).asnumpy()
            pred = pred.asnumpy()
            if pred.shape != label.shape:
                pred = pred.argmax(axis=1)
            else:
                pred = pred.astype('int32', copy=False)
            n = max(pred.max(), label.max())
            if n >= self.k:
                self._grow(n + 1 - self.k)
            bcm = numpy.zeros((self.k, self.k))
            for i, j in zip(pred, label):
                bcm[i, j] += 1
            self.lcm += bcm
            self.gcm += bcm
        self.num_inst += 1
        self.global_num_inst += 1

    @property
    def sum_metric(self):
        return self._calc_mcc(self.lcm) * self.num_inst

    @property
    def global_sum_metric(self):
        return self._calc_mcc(self.gcm) * self.global_num_inst

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.global_num_inst = 0.0
        self.gcm = numpy.zeros((self.k, self.k))
        self.reset_local()

    def reset_local(self):
        """Resets the local portion of the internal evaluation results to initial state."""
        self.num_inst = 0.0
        self.lcm = numpy.zeros((self.k, self.k))