import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
@alias('pearsonr')
class PearsonCorrelation(EvalMetric):
    """Computes Pearson correlation.

    The pearson correlation is given by

    .. math::
        \\frac{cov(y, \\hat{y})}{\\sigma{y}\\sigma{\\hat{y}}}

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
    average : str, default 'macro'
        Strategy to be used for aggregating across mini-batches.
            "macro": average the pearsonr scores for each batch.
            "micro": compute a single pearsonr score across all batches.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([[1, 0], [0, 1], [0, 1]])]
    >>> pr = mx.metric.PearsonCorrelation()
    >>> pr.update(labels, predicts)
    >>> print pr.get()
    ('pearsonr', 0.42163704544016178)
    """

    def __init__(self, name='pearsonr', output_names=None, label_names=None, average='macro'):
        self.average = average
        super(PearsonCorrelation, self).__init__(name, output_names=output_names, label_names=label_names, has_global_stats=True)
        if self.average == 'micro':
            self.reset_micro()

    def reset_micro(self):
        self._sse_p = 0
        self._mean_p = 0
        self._sse_l = 0
        self._mean_l = 0
        self._pred_nums = 0
        self._label_nums = 0
        self._conv = 0

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0
        if self.average == 'micro':
            self.reset_micro()

    def update_variance(self, new_values, *aggregate):
        count, mean, m_2 = aggregate
        count += len(new_values)
        delta = new_values - mean
        mean += numpy.sum(delta / count)
        delta_2 = new_values - mean
        m_2 += numpy.sum(delta * delta_2)
        return (count, mean, m_2)

    def update_cov(self, label, pred):
        self._conv = self._conv + numpy.sum((label - self._mean_l) * (pred - self._mean_p))

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
            check_label_shapes(label, pred, False, True)
            label = label.asnumpy().ravel().astype(numpy.float64)
            pred = pred.asnumpy().ravel().astype(numpy.float64)
            if self.average == 'macro':
                pearson_corr = numpy.corrcoef(pred, label)[0, 1]
                self.sum_metric += pearson_corr
                self.global_sum_metric += pearson_corr
                self.num_inst += 1
                self.global_num_inst += 1
            else:
                self.global_num_inst += 1
                self.num_inst += 1
                self._label_nums, self._mean_l, self._sse_l = self.update_variance(label, self._label_nums, self._mean_l, self._sse_l)
                self.update_cov(label, pred)
                self._pred_nums, self._mean_p, self._sse_p = self.update_variance(pred, self._pred_nums, self._mean_p, self._sse_p)

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        if self.average == 'macro':
            return (self.name, self.sum_metric / self.num_inst)
        else:
            n = self._label_nums
            pearsonr = self._conv / ((n - 1) * numpy.sqrt(self._sse_p / (n - 1)) * numpy.sqrt(self._sse_l / (n - 1)))
            return (self.name, pearsonr)