from itertools import chain
from functools import lru_cache
import torch as th
import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
def _dummy_batch(self, bsz, maxlen):
    batch = super()._dummy_batch(bsz, maxlen)
    batch['know_vec'] = th.zeros(bsz, 2, 2).long().cuda()
    ck_mask = (th.ones(bsz, 2, dtype=th.uint8) != 0).cuda()
    batch['ck_mask'] = ck_mask
    batch['cs_ids'] = th.zeros(bsz).long().cuda()
    batch['use_cs_ids'] = True
    return batch