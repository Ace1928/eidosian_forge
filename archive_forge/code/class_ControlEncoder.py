import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
class ControlEncoder(nn.Module):
    """
    Given CT control variable, gives the concatenated control embeddings vector.
    """

    def __init__(self, control_settings):
        super().__init__()
        self.control_settings = control_settings
        self.idx2ctrl = {d['idx']: control for control, d in self.control_settings.items()}
        self.control_embeddings = nn.ModuleDict({c: nn.Embedding(d['num_buckets'], d['embsize'], sparse=False) for c, d in control_settings.items()})

    def forward(self, control_inputs):
        """
        Forward pass.

        :param control_inputs: (bsz x num_control_vars) LongTensor of control
            variable values (i.e. bucket ids)

        :returns: control_embs, (bsz x sum of control emb sizes) FloatTensor of
            control variable embeddings, concatenated
        """
        control_inputs = torch.split(control_inputs, 1, dim=1)
        control_inputs = [torch.squeeze(t, 1) for t in control_inputs]
        assert len(control_inputs) == len(self.control_settings)
        control_embs = []
        for idx, inputs in enumerate(control_inputs):
            control = self.idx2ctrl[idx]
            control_embs.append(self.control_embeddings[control](inputs))
        control_embs = torch.cat(control_embs, dim=1)
        return control_embs