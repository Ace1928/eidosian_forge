import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
def _decode_forced(self, ys, ctrl_inputs, encoder_states):
    """
        Decode with teacher forcing.
        """
    bsz = ys.size(0)
    seqlen = ys.size(1)
    hidden = encoder_states[1]
    attn_params = (encoder_states[0], encoder_states[2])
    y_in = ys.narrow(1, 0, seqlen - 1)
    xs = torch.cat([self._starts(bsz), y_in], 1)
    scores = []
    if self.attn_type == 'none':
        output, hidden = self.decoder(xs, ctrl_inputs, hidden, attn_params)
        score = self.output(output)
        scores.append(score)
    else:
        for i in range(seqlen):
            xi = xs.select(1, i).unsqueeze(1)
            output, hidden = self.decoder(xi, ctrl_inputs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
    scores = torch.cat(scores, 1)
    return scores