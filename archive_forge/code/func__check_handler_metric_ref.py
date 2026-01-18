from ...loss import SoftmaxCrossEntropyLoss
from ....metric import Accuracy, EvalMetric, CompositeEvalMetric
def _check_handler_metric_ref(handler, known_metrics):
    for attribute in dir(handler):
        if any((keyword in attribute for keyword in ['metric' or 'monitor'])):
            reference = getattr(handler, attribute)
            if not reference:
                continue
            elif isinstance(reference, list):
                for metric in reference:
                    _check_metric_known(handler, metric, known_metrics)
            else:
                _check_metric_known(handler, reference, known_metrics)