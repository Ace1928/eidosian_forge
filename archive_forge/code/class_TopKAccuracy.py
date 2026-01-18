import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@register
@alias('top_k_accuracy', 'top_k_acc')
class TopKAccuracy(EvalMetric):
    """Computes top k predictions accuracy.

    `TopKAccuracy` differs from Accuracy in that it considers the prediction
    to be ``True`` as long as the ground truth label is in the top K
    predicated labels.

    If `top_k` = ``1``, then `TopKAccuracy` is identical to `Accuracy`.

    Parameters
    ----------
    top_k : int
        Whether targets are in top k predictions.
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
    >>> np.random.seed(999)
    >>> top_k = 3
    >>> labels = [mx.nd.array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    >>> predicts = [mx.nd.array(np.random.rand(10, 10))]
    >>> acc = mx.metric.TopKAccuracy(top_k=top_k)
    >>> acc.update(labels, predicts)
    >>> print acc.get()
    ('top_k_accuracy', 0.3)
    """

    def __init__(self, top_k=1, name='top_k_accuracy', output_names=None, label_names=None):
        super(TopKAccuracy, self).__init__(name, top_k=top_k, output_names=output_names, label_names=label_names, has_global_stats=True)
        self.top_k = top_k
        assert self.top_k > 1, 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k

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
        for label, pred_label in zip(labels, preds):
            assert len(pred_label.shape) <= 2, 'Predictions should be no more than 2 dims'
            pred_label = numpy.argpartition(pred_label.asnumpy().astype('float32'), -self.top_k)
            label = label.asnumpy().astype('int32')
            check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    num_correct = (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
                    self.sum_metric += num_correct
                    self.global_sum_metric += num_correct
            self.num_inst += num_samples
            self.global_num_inst += num_samples