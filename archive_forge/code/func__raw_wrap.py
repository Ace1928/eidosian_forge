import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
@classmethod
def _raw_wrap(cls, values, layout, sparse_dot_sdd, sparse_dot_dsd, sparse_softmax):
    matrix = cls.__new__(cls, values, layout)
    matrix.__values = values
    matrix.__layout = layout
    matrix.__sparse_dot_sdd = sparse_dot_sdd
    matrix.__sparse_dot_dsd = sparse_dot_dsd
    matrix.__sparse_softmax = sparse_softmax
    return matrix