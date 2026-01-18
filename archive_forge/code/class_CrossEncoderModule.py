from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .transformer import TransformerRankerAgent
from parlai.utils.torch import concat_without_padding
import torch
class CrossEncoderModule(torch.nn.Module):
    """
    A simple wrapper around the transformer encoder which adds a linear layer.
    """

    def __init__(self, opt, dict, null_idx):
        super(CrossEncoderModule, self).__init__()
        n_positions = get_n_positions_from_options(opt)
        embeddings = torch.nn.Embedding(len(dict), opt['embedding_size'], padding_idx=null_idx)
        torch.nn.init.normal_(embeddings.weight, 0, opt['embedding_size'] ** (-0.5))
        self.encoder = TransformerEncoder(n_heads=opt['n_heads'], n_layers=opt['n_layers'], embedding_size=opt['embedding_size'], ffn_size=opt['ffn_size'], vocabulary_size=len(dict), embedding=embeddings, dropout=opt['dropout'], attention_dropout=opt['attention_dropout'], relu_dropout=opt['relu_dropout'], padding_idx=null_idx, learn_positional_embeddings=opt['learn_positional_embeddings'], embeddings_scale=opt['embeddings_scale'], reduction_type=opt.get('reduction_type', 'first'), n_positions=n_positions, n_segments=2, activation=opt['activation'], variant=opt['variant'], output_scaling=opt['output_scaling'])
        self.linear_layer = torch.nn.Linear(opt['embedding_size'], 1)

    def forward(self, tokens, segments):
        """
        Scores each concatenation text + candidate.
        """
        encoded = self.encoder(tokens, None, segments)
        res = self.linear_layer(encoded)
        return res