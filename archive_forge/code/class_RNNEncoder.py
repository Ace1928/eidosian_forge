import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
class RNNEncoder(nn.Module):
    """
    RNN Encoder.
    """

    def __init__(self, num_features, embeddingsize, hiddensize, padding_idx=0, rnn_class='lstm', numlayers=2, dropout=0.1, bidirectional=False, shared_lt=None, shared_rnn=None, input_dropout=0, unknown_idx=None, sparse=False):
        """
        Initialize recurrent encoder.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hiddensize
        if input_dropout > 0 and unknown_idx is None:
            raise RuntimeError('input_dropout > 0 but unknown_idx not set')
        self.input_dropout = UnknownDropout(unknown_idx, input_dropout)
        if shared_lt is None:
            self.lt = nn.Embedding(num_features, embeddingsize, padding_idx=padding_idx, sparse=sparse)
        else:
            self.lt = shared_lt
        if shared_rnn is None:
            self.rnn = rnn_class(embeddingsize, hiddensize, numlayers, dropout=dropout if numlayers > 1 else 0, batch_first=True, bidirectional=bidirectional)
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

    def forward(self, xs):
        """
        Encode sequence.

        :param xs: (bsz x seqlen) LongTensor of input token indices

        :returns: encoder outputs, hidden state, attention mask
            encoder outputs are the output state at each step of the encoding.
            the hidden state is the final hidden state of the encoder.
            the attention mask is a mask of which input values are nonzero.
        """
        bsz = len(xs)
        xs = self.input_dropout(xs)
        xes = self.dropout(self.lt(xs))
        attn_mask = xs.ne(0)
        try:
            x_lens = torch.sum(attn_mask.int(), dim=1)
            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            packed = True
        except ValueError:
            packed = False
        encoder_output, hidden = self.rnn(xes)
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True)
        if self.dirs > 1:
            if isinstance(self.rnn, nn.LSTM):
                hidden = (hidden[0].view(-1, self.dirs, bsz, self.hsz).sum(1), hidden[1].view(-1, self.dirs, bsz, self.hsz).sum(1))
            else:
                hidden = hidden.view(-1, self.dirs, bsz, self.hsz).sum(1)
        return (encoder_output, hidden, attn_mask)