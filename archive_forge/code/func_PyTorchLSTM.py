from functools import partial
from typing import Callable, Optional, Tuple, cast
from ..backends import Ops
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d, Floats4d, Padded, Ragged
from ..util import get_width
from .noop import noop
@registry.layers('PyTorchLSTM.v1')
def PyTorchLSTM(nO: int, nI: int, *, bi: bool=False, depth: int=1, dropout: float=0.0) -> Model[Padded, Padded]:
    import torch.nn
    from .pytorchwrapper import PyTorchRNNWrapper
    from .with_padded import with_padded
    if depth == 0:
        return noop()
    nH = nO
    if bi:
        nH = nO // 2
    pytorch_rnn = PyTorchRNNWrapper(torch.nn.LSTM(nI, nH, depth, bidirectional=bi, dropout=dropout))
    pytorch_rnn.set_dim('nO', nO)
    pytorch_rnn.set_dim('nI', nI)
    return with_padded(pytorch_rnn)