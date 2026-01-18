import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
def _align_inds(self, encoder_states, cand_inds):
    """
        Select the encoder states relevant to valid candidates.
        """
    enc_out, hidden, attn_mask = encoder_states
    if isinstance(hidden, torch.Tensor):
        hid, cell = (hidden, None)
    else:
        hid, cell = hidden
    if len(cand_inds) != hid.size(1):
        cand_indices = hid.new(cand_inds)
        hid = hid.index_select(1, cand_indices)
        if cell is None:
            hidden = hid
        else:
            cell = cell.index_select(1, cand_indices)
            hidden = (hid, cell)
        if self.attn_type != 'none':
            enc_out = enc_out.index_select(0, cand_indices)
            attn_mask = attn_mask.index_select(0, cand_indices)
    return (enc_out, hidden, attn_mask)