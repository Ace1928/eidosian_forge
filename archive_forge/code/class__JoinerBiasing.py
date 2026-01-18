import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
class _JoinerBiasing(torch.nn.Module):
    """Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
        biasing (bool): perform biasing
        deepbiasing (bool): perform deep biasing
        attndim (int): dimension of the biasing vector hptr

    """

    def __init__(self, input_dim: int, output_dim: int, activation: str='relu', biasing: bool=False, deepbiasing: bool=False, attndim: int=1) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.biasing = biasing
        self.deepbiasing = deepbiasing
        if self.biasing and self.deepbiasing:
            self.biasinglinear = torch.nn.Linear(attndim, input_dim, bias=True)
            self.attndim = attndim
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f'Unsupported activation {activation}')

    def forward(self, source_encodings: torch.Tensor, source_lengths: torch.Tensor, target_encodings: torch.Tensor, target_lengths: torch.Tensor, hptr: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
            hptr (torch.Tensor): deep biasing vector with shape `(B, T, U, A)`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                torch.Tensor
                    joint network second last layer output (i.e. before self.linear), with shape `(B, T, U, D)`.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        if self.biasing and self.deepbiasing and (hptr is not None):
            hptr = self.biasinglinear(hptr)
            joint_encodings += hptr
        elif self.biasing and self.deepbiasing:
            joint_encodings += self.biasinglinear(joint_encodings.new_zeros(1, self.attndim)).mean() * 0
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return (output, source_lengths, target_lengths, activation_out)