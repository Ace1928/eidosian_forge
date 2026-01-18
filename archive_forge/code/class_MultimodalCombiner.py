import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
class MultimodalCombiner(nn.Module):
    """
    Multimodal Combination module.
    """

    def __init__(self, n_heads, n_layers, hidden_dim, ffn_size, reduction=True, attention_dropout=0.0, relu_dropout=0.0, learn_positional_embeddings=False):
        super().__init__()
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.out_dim = hidden_dim
        self.dim = hidden_dim
        self.reduction = reduction
        assert hidden_dim % n_heads == 0, 'MM-Combiner dim must be multiple of n_heads'
        n_positions = 1024
        self.position_embeddings = nn.Embedding(n_positions, hidden_dim)
        if not learn_positional_embeddings:
            create_position_codes(n_positions, hidden_dim, out=self.position_embeddings.weight)
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, hidden_dim ** (-0.5))
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(n_heads, hidden_dim, ffn_size, attention_dropout, relu_dropout))

    def forward(self, tensor, mask):
        """
        Forward pass.

        :param tensor:
            a [bsz, seq_len, hidden_dim] FloatTensor
        :param mask:
            a [bsz, seq_len] ByteTensor filled with 1 when inside the sequence and 0 outside.

        :return:
            output: a [bsz, hidden_dim] FloatTensor of encodings
            mask: the same as before
        """
        seq_len = tensor.size(1)
        positions = tensor.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor *= mask.unsqueeze(-1).float()
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)
        if self.reduction:
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-20)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return (output, mask)