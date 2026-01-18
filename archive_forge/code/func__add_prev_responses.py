import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.torch import padded_tensor
from parlai.agents.transformer.transformer import TransformerRankerAgent
from .feedback_classifier.feedback_classifier import FeedbackClassifierRegex
from .modules import SelfFeedingModel
def _add_prev_responses(self, batch, cands, cand_vecs, label_inds, source):
    assert source not in ['fixed', 'vocab']
    self._extract_prev_responses(batch)
    prev_cands = self.prev_responses
    prev_cand_vecs = [torch.Tensor(self.dict.txt2vec(cand)) for cand in prev_cands]
    if source == 'batch':
        cands += prev_cands
        cand_vecs, _ = padded_tensor([vec for vec in cand_vecs] + prev_cand_vecs, use_cuda=self.use_cuda)
    elif source == 'inline':
        raise NotImplementedError
    return (cands, cand_vecs)