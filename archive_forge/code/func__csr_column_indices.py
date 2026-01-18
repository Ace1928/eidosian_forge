import torch
from xformers.ops import masked_matmul
from xformers.sparse import _csr_ops
from xformers.sparse.utils import (
@property
def _csr_column_indices(self):
    return self.__column_indices