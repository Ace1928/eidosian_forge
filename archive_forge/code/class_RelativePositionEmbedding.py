from typing import Union
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.utils.typing import TensorType
class RelativePositionEmbedding(nn.Module):
    """Creates a [seq_length x seq_length] matrix for rel. pos encoding.

    Denoted as Phi in [2] and [3]. Phi is the standard sinusoid encoding
    matrix.

    Args:
        seq_length: The max. sequence length (time axis).
        out_dim: The number of nodes to go into the first Tranformer
            layer with.

    Returns:
        torch.Tensor: The encoding matrix Phi.
    """

    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        out_range = torch.arange(0, self.out_dim, 2.0)
        inverse_freq = 1 / 10000 ** (out_range / self.out_dim)
        self.register_buffer('inverse_freq', inverse_freq)

    def forward(self, seq_length):
        pos_input = torch.arange(seq_length - 1, -1, -1.0, dtype=torch.float).to(self.inverse_freq.device)
        sinusoid_input = torch.einsum('i,j->ij', pos_input, self.inverse_freq)
        pos_embeddings = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)
        return pos_embeddings[:, None, :]