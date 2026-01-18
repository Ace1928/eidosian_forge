import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
class MCC(EvalMetric):
    """Computes the Matthews Correlation Coefficient of a binary classification problem.

    While slower to compute than F1 the MCC can give insight that F1 or Accuracy cannot.
    For instance, if the network always predicts the same result
    then the MCC will immeadiately show this. The MCC is also symetric with respect
    to positive and negative categorization, however, there needs to be both
    positive and negative examples in the labels or it will always return 0.
    MCC of 0 is uncorrelated, 1 is completely correlated, and -1 is negatively correlated.

    .. math::
        \\text{MCC} = \\frac{ TP \\times TN - FP \\times FN }
        {\\sqrt{ (TP + FP) ( TP + FN ) ( TN + FP ) ( TN + FN ) } }

    where 0 terms in the denominator are replaced by 1.

    .. note::

        This version of MCC only supports binary classification.  See PCC.

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
            "macro": average the MCC for each batch.
            "micro": compute a single MCC across all batches.

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
        [0.]*(false_positives + true_negatives) +
        [1.]*(false_negatives + true_positives)
    )]
    >>> f1 = mx.metric.F1()
    >>> f1.update(preds = predicts, labels = labels)
    >>> mcc = mx.metric.MCC()
    >>> mcc.update(preds = predicts, labels = labels)
    >>> print f1.get()
    ('f1', 0.95233560306652054)
    >>> print mcc.get()
    ('mcc', 0.01917751877733392)
    """

    def __init__(self, name='mcc', output_names=None, label_names=None, average='macro'):
        self._average = average
        self._metrics = _BinaryClassificationMetrics()
        EvalMetric.__init__(self, name=name, output_names=output_names, label_names=label_names, has_global_stats=True)

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
            self._metrics.update_binary_stats(label, pred)
        if self._average == 'macro':
            self.sum_metric += self._metrics.matthewscc()
            self.global_sum_metric += self._metrics.matthewscc(use_global=True)
            self.num_inst += 1
            self.global_num_inst += 1
            self._metrics.reset_stats()
        else:
            self.sum_metric = self._metrics.matthewscc() * self._metrics.total_examples
            self.global_sum_metric = self._metrics.matthewscc(use_global=True) * self._metrics.global_total_examples
            self.num_inst = self._metrics.total_examples
            self.global_num_inst = self._metrics.global_total_examples

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.sum_metric = 0.0
        self.num_inst = 0.0
        self.global_sum_metric = 0.0
        self.global_num_inst = 0.0
        self._metrics.reset_stats()

    def reset_local(self):
        """Resets the internal evaluation result to initial state."""
        self.sum_metric = 0.0
        self.num_inst = 0.0
        self._metrics.local_reset_stats()