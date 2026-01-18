import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
class AttentionLayer(nn.Module):
    """
    Computes attention between hidden and encoder states.

    See arxiv.org/abs/1508.04025 for more info on each attention type.
    """

    def __init__(self, attn_type, hiddensize, embeddingsize, bidirectional=False, attn_length=-1, attn_time='pre'):
        """
        Initialize attention layer.
        """
        super().__init__()
        self.attention = attn_type
        if self.attention != 'none':
            hsz = hiddensize
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                input_dim = embeddingsize
            elif attn_time == 'post':
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim, bias=False)
            if self.attention == 'local':
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, attn_params):
        """
        Compute attention over attn_params given input and hidden states.

        :param xes:         input state. will be combined with applied
                            attention.
        :param hidden:      hidden state from model. will be used to select
                            states to attend to in from the attn_params.
        :param attn_params: tuple of encoder output states and a mask showing
                            which input indices are nonzero.

        :returns: output, attn_weights
                  output is a new state of same size as input state `xes`.
                  attn_weights are the weights given to each state in the
                  encoder outputs.
        """
        if self.attention == 'none':
            return (xes, None)
        if type(hidden) == tuple:
            hidden = hidden[0]
        last_hidden = hidden[-1]
        enc_out, attn_mask = attn_params
        bsz, seqlen, hszXnumdir = enc_out.size()
        numlayersXnumdir = last_hidden.size(1)
        if self.attention == 'local':
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)
            if seqlen > self.max_length:
                offset = seqlen - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
                seqlen = self.max_length
            if attn_weights.size(1) > seqlen:
                attn_weights = attn_weights.narrow(1, 0, seqlen)
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                hid = hid.expand(bsz, seqlen, numlayersXnumdir)
                h_merged = torch.cat((enc_out, hid), 2)
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                if numlayersXnumdir != hszXnumdir:
                    hid = torch.cat([hid, hid], 2)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)
            elif self.attention == 'general':
                hid = self.attn(hid)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)
            if attn_mask is not None:
                attn_w_premask.masked_fill_(~attn_mask, -NEAR_INF)
            attn_weights = F.softmax(attn_w_premask, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        output = torch.tanh(self.attn_combine(merged).unsqueeze(1))
        return (output, attn_weights)