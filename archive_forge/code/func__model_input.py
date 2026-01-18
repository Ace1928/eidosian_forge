from itertools import chain
from functools import lru_cache
import torch as th
import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
def _model_input(self, batch):
    return (batch.text_vec, batch.know_vec, batch.ck_mask, batch.cs_ids, batch.use_cs_ids)