from typing import Dict, Any
from abc import abstractmethod
from itertools import islice
import os
from tqdm import tqdm
import random
import torch
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import warn_once
from parlai.utils.torch import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
def _set_label_cands_vec(self, *args, **kwargs):
    """
        Set the 'label_candidates_vec' field in the observation.

        Useful to override to change vectorization behavior.
        """
    obs = args[0]
    if 'labels' in obs:
        cands_key = 'candidates'
    else:
        cands_key = 'eval_candidates'
    if self.opt[cands_key] not in ['inline', 'batch-all-cands']:
        return obs
    return super()._set_label_cands_vec(*args, **kwargs)