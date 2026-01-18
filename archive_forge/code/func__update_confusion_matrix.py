from parlai.core.opt import Opt
from parlai.utils.torch import PipelineHelper
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.core.metrics import Metric, AverageMetric
from typing import List, Optional, Tuple, Dict
from parlai.utils.typing import TScalar
import parlai.utils.logging as logging
import torch
import torch.nn.functional as F
def _update_confusion_matrix(self, batch, predictions):
    """
        Update the confusion matrix given the batch and predictions.

        :param predictions:
            (list of string of length batchsize) label predicted by the
            classifier
        :param batch:
            a Batch object (defined in torch_agent.py)
        """
    f1_dict = {}
    for class_name in self.class_list:
        prec_str = f'class_{class_name}_prec'
        recall_str = f'class_{class_name}_recall'
        f1_str = f'class_{class_name}_f1'
        precision, recall, f1 = ConfusionMatrixMetric.compute_metrics(predictions, batch.labels, class_name)
        f1_dict[class_name] = f1
        self.record_local_metric(prec_str, precision)
        self.record_local_metric(recall_str, recall)
        self.record_local_metric(f1_str, f1)
    self.record_local_metric('weighted_f1', WeightedF1Metric.compute_many(f1_dict))