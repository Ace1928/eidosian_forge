import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
def _extract_cur(self, encoder_states, index, num_cands):
    """
        Extract encoder states at current index and expand them.
        """
    enc_out, hidden, attn_mask = encoder_states
    if isinstance(hidden, torch.Tensor):
        cur_hid = hidden.select(1, index).unsqueeze(1).expand(-1, num_cands, -1)
    else:
        cur_hid = (hidden[0].select(1, index).unsqueeze(1).expand(-1, num_cands, -1).contiguous(), hidden[1].select(1, index).unsqueeze(1).expand(-1, num_cands, -1).contiguous())
    cur_enc, cur_mask = (None, None)
    if self.attn_type != 'none':
        cur_enc = enc_out[index].unsqueeze(0).expand(num_cands, -1, -1)
        cur_mask = attn_mask[index].unsqueeze(0).expand(num_cands, -1)
    return (cur_enc, cur_hid, cur_mask)