from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import padded_tensor, argsort, neginf
from .modules import Seq2seq, opt_to_kwargs
from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, Counter
from operator import attrgetter
import os
import math
import json
import tempfile
import copy
def _build_cands(self, batch):
    if not batch.candidates:
        return (None, None)
    cand_inds = [i for i in range(len(batch.candidates)) if batch.candidates[i]]
    cands = [batch.candidate_vecs[i] for i in cand_inds]
    max_cands_len = max([max([cand.size(0) for cand in cands_i]) for cands_i in cands])
    for i, c in enumerate(cands):
        cands[i] = padded_tensor(c, use_cuda=self.use_cuda, max_len=max_cands_len)[0].unsqueeze(0)
    cands = torch.cat(cands, 0)
    return (cands, cand_inds)