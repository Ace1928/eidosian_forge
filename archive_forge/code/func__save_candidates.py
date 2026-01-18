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
def _save_candidates(self, vecs, path, cand_type='vectors'):
    """
        Save cached vectors.
        """
    logging.info(f'Saving fixed candidate set {cand_type} to {path}')
    with open(path, 'wb') as f:
        torch.save(vecs, f)