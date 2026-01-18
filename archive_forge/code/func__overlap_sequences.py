import math
import torch
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.numpy import pad
def _overlap_sequences(x, sequence_stride):
    x = convert_to_tensor(x)
    *batch_shape, num_sequences, sequence_length = x.shape
    if sequence_stride > sequence_length:
        raise ValueError(f'`sequence_stride` must equal or less than x.shape[-1]. Received: sequence_stride={sequence_stride}, x.shape[-1]={sequence_length}')
    if sequence_stride < sequence_length / num_sequences:
        raise ValueError(f'`sequence_stride` must equal or greater than x.shape[-1] / x.shape[-2]. Received: sequence_stride={sequence_stride}, x.shape[-1]={sequence_length}, x.shape[-2]={num_sequences}')
    flat_batchsize = math.prod(batch_shape)
    x = torch.reshape(x, (flat_batchsize, num_sequences, sequence_length))
    output_size = sequence_stride * (num_sequences - 1) + sequence_length
    nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride
    padded_segment_len = nstep_per_segment * sequence_stride
    x = torch.nn.functional.pad(x, (0, padded_segment_len - sequence_length, 0, 0, 0, 0))
    x = torch.reshape(x, (flat_batchsize, num_sequences, nstep_per_segment, sequence_stride))
    x = torch.permute(x, (0, 2, 1, 3))
    x = torch.nn.functional.pad(x, (0, 0, 0, num_sequences, 0, 0, 0, 0))
    shrinked = x.shape[2] - 1
    x = torch.reshape(x, (flat_batchsize, -1))
    x = x[:, :nstep_per_segment * shrinked * sequence_stride]
    x = torch.reshape(x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride))
    x = torch.sum(x, dim=1)[:, :output_size]
    return torch.reshape(x, tuple(batch_shape) + (-1,))