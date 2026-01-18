import torch
import torch.ao.nn.quantizable
from torch.nn.modules.pooling import MaxPool2d
from .activation import ReLU6, Hardswish, ELU, LeakyReLU, Sigmoid, Softmax, MultiheadAttention, PReLU
from .dropout import Dropout
from .batchnorm import BatchNorm2d, BatchNorm3d
from .normalization import LayerNorm, GroupNorm, InstanceNorm1d, \
from .conv import Conv1d, Conv2d, Conv3d
from .conv import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from .linear import Linear
from .embedding_ops import Embedding, EmbeddingBag
from .rnn import LSTM
from .functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
class DeQuantize(torch.nn.Module):
    """Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod):
        return DeQuantize()