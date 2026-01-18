from ...loss import SoftmaxCrossEntropyLoss
from ....metric import Accuracy, EvalMetric, CompositeEvalMetric
def _check_metric_known(handler, metric, known_metrics):
    if metric not in known_metrics:
        raise ValueError('Event handler {} refers to a metric instance {} outside of the known training and validation metrics. Please use the metrics from estimator.train_metrics and estimator.val_metrics instead.'.format(type(handler).__name__, metric))