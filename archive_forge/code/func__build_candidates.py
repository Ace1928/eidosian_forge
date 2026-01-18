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
def _build_candidates(self, batch, source, mode):
    cands, cand_vecs, label_inds = super()._build_candidates(batch, source, mode)
    if self.opt['prev_response_negatives'] and mode == 'train':
        cands, cand_vecs = self._add_prev_responses(batch, cands, cand_vecs, label_inds, source)
    return (cands, cand_vecs, label_inds)