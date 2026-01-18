import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def _forward_padded(self, x, x_mask):
    """
        Slower (significantly), but more precise, encoding that handles padding.
        """
    lengths = x_mask.data.eq(0).long().sum(1).squeeze()
    _, idx_sort = torch.sort(lengths, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    lengths = list(lengths[idx_sort])
    idx_sort = Variable(idx_sort)
    idx_unsort = Variable(idx_unsort)
    x = x.index_select(0, idx_sort)
    x = x.transpose(0, 1).contiguous()
    rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
    outputs = [rnn_input]
    for i in range(self.num_layers):
        rnn_input = outputs[-1]
        if self.dropout_rate > 0:
            dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
            rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
        outputs.append(self.rnns[i](rnn_input)[0])
    for i, o in enumerate(outputs[1:], 1):
        outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]
    if self.concat_layers:
        output = torch.cat(outputs[1:], 2)
    else:
        output = outputs[-1]
    output = output.transpose(0, 1).contiguous()
    output = output.index_select(0, idx_unsort)
    if self.dropout_output and self.dropout_rate > 0:
        output = F.dropout(output, p=self.dropout_rate, training=self.training)
    return output