import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
class F1(EvalMetric):
    """Computes the F1 score of a binary classification problem.

    The F1 score is equivalent to harmonic mean of the precision and recall,
    where the best value is 1.0 and the worst value is 0.0. The formula for F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    The formula for precision and recall is::

        precision = true_positives / (true_positives + false_positives)
        recall    = true_positives / (true_positives + false_negatives)

    .. note::

        This F1 score only supports binary classification.

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
            "macro": average the F1 scores for each batch.
            "micro": compute a single F1 score across all batches.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0., 1., 1.])]
    >>> f1 = mx.metric.F1()
    >>> f1.update(preds = predicts, labels = labels)
    >>> print f1.get()
    ('f1', 0.8)
    """

    def __init__(self, name='f1', output_names=None, label_names=None, average='macro'):
        self.average = average
        self.metrics = _BinaryClassificationMetrics()
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
            self.metrics.update_binary_stats(label, pred)
        if self.average == 'macro':
            self.sum_metric += self.metrics.fscore
            self.global_sum_metric += self.metrics.global_fscore
            self.num_inst += 1
            self.global_num_inst += 1
            self.metrics.reset_stats()
        else:
            self.sum_metric = self.metrics.fscore * self.metrics.total_examples
            self.global_sum_metric = self.metrics.global_fscore * self.metrics.global_total_examples
            self.num_inst = self.metrics.total_examples
            self.global_num_inst = self.metrics.global_total_examples

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.sum_metric = 0.0
        self.num_inst = 0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0
        self.metrics.reset_stats()

    def reset_local(self):
        """Resets the internal evaluation result to initial state."""
        self.sum_metric = 0.0
        self.num_inst = 0
        self.metrics.local_reset_stats()