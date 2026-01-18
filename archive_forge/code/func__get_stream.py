import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _get_stream(self, stream):
    if stream is None:
        stream = cupy.cuda.stream.get_current_stream()
    return stream.ptr