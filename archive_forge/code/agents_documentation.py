from itertools import chain
from functools import lru_cache
import torch as th
import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE

        Wizard custom batchify, which passes along the knowledge/title.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        