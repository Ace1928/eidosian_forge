import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
class MatDescriptor(object):

    def __init__(self, descriptor):
        self.descriptor = descriptor

    @classmethod
    def create(cls):
        descr = _cusparse.createMatDescr()
        return MatDescriptor(descr)

    def __reduce__(self):
        return (self.create, ())

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        if is_shutting_down():
            return
        if self.descriptor:
            _cusparse.destroyMatDescr(self.descriptor)
            self.descriptor = None

    def set_mat_type(self, typ):
        _cusparse.setMatType(self.descriptor, typ)

    def set_mat_index_base(self, base):
        _cusparse.setMatIndexBase(self.descriptor, base)

    def set_mat_fill_mode(self, fill_mode):
        _cusparse.setMatFillMode(self.descriptor, fill_mode)

    def set_mat_diag_type(self, diag_type):
        _cusparse.setMatDiagType(self.descriptor, diag_type)