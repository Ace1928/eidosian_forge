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
def _format_interactive_output(self, probs, prediction_id):
    """
        Format interactive mode output with scores.
        """
    preds = []
    for i, pred_id in enumerate(prediction_id.tolist()):
        prob = round_sigfigs(probs[i][pred_id], 4)
        preds.append('Predicted class: {}\nwith probability: {}'.format(self.class_list[pred_id], prob))
    return preds