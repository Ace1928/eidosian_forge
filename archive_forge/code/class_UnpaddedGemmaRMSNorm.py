from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gemma.configuration_gemma import GemmaConfig
class UnpaddedGemmaRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps):
        """
        UnpaddedGemmaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = torch.tensor(eps, dtype=torch.get_default_dtype())
        global RMS_NORM_TRACED
        if RMS_NORM_TRACED is None:
            RMS_NORM_TRACED = torch.jit.trace(rms_norm, (torch.ones(hidden_size), torch.ones(hidden_size), self.variance_epsilon))

    def forward(self, hidden_states):
        global RMS_NORM_TRACED
        return RMS_NORM_TRACED(hidden_states, self.weight, self.variance_epsilon)