import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .modeling_base import PreTrainedModelWrapper
class ValueHead(nn.Module):
    """
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, 'summary_dropout_prob'):
            summary_dropout_prob = kwargs.pop('summary_dropout_prob', 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        if hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        if hasattr(config, 'word_embed_proj_dim'):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, 'is_encoder_decoder'):
            if config.is_encoder_decoder and hasattr(config, 'decoder'):
                if hasattr(config.decoder, 'hidden_size'):
                    hidden_size = config.decoder.hidden_size
        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output