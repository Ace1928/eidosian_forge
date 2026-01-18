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
def _extract_prev_responses(self, batch):
    warn_once('WARNING: This code is specific to self-feeding formatted examples')
    p1 = self.dict.txt2vec('__p1__')[0]
    p2 = self.dict.txt2vec('__p2__')[0]
    self.prev_responses = []
    for text_vec in batch.text_vec:
        p1s = (text_vec == p1).nonzero()
        p2s = (text_vec == p2).nonzero()
        if len(p1s) and len(p2s):
            response_vec = text_vec[p2s[-1] + 1:p1s[-1]]
        else:
            response_vec = [self.NULL_IDX]
        response = self.dict.vec2txt(response_vec)
        self.prev_responses.append(response)