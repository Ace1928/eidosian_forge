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
class PrecisionMetric(ConfusionMatrixMetric):
    """
    Class that takes in a ConfusionMatrixMetric and computes precision for classifier.
    """

    def value(self) -> float:
        if self._true_positives == 0:
            return 0.0
        else:
            return self._true_positives / (self._true_positives + self._false_positives)