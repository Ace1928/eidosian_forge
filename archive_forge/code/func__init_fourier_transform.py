import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...utils import is_scipy_available
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_fnet import FNetConfig
def _init_fourier_transform(self, config):
    if not config.use_tpu_fourier_optimizations:
        self.fourier_transform = partial(torch.fft.fftn, dim=(1, 2))
    elif config.max_position_embeddings <= 4096:
        if is_scipy_available():
            self.register_buffer('dft_mat_hidden', torch.tensor(linalg.dft(config.hidden_size), dtype=torch.complex64))
            self.register_buffer('dft_mat_seq', torch.tensor(linalg.dft(config.tpu_short_seq_length), dtype=torch.complex64))
            self.fourier_transform = partial(two_dim_matmul, matrix_dim_one=self.dft_mat_seq, matrix_dim_two=self.dft_mat_hidden)
        else:
            logging.warning('SciPy is needed for DFT matrix calculation and is not found. Using TPU optimized fast fourier transform instead.')
            self.fourier_transform = fftn
    else:
        self.fourier_transform = fftn