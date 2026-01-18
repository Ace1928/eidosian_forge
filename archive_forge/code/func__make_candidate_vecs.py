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
def _make_candidate_vecs(self, cands):
    """
        Prebuild cached vectors for fixed candidates.
        """
    cand_batches = [cands[i:i + 512] for i in range(0, len(cands), 512)]
    logging.info(f'Vectorizing fixed candidate set ({len(cand_batches)} batch(es) of up to 512)')
    cand_vecs = []
    for batch in tqdm(cand_batches):
        cand_vecs.extend(self.vectorize_fixed_candidates(batch))
    return padded_3d([cand_vecs], pad_idx=self.NULL_IDX, dtype=cand_vecs[0].dtype).squeeze(0)