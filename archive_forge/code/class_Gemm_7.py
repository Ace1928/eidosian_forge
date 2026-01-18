import numpy as np
from onnx.reference.op_run import OpRun
class Gemm_7(OpRun):

    def _run(self, a, b, c=None, alpha=None, beta=None, transA=None, transB=None):
        if transA:
            _meth = _gemm11 if transB else _gemm10
        else:
            _meth = _gemm01 if transB else _gemm00
        return (_meth(a, b, c, alpha, beta).astype(a.dtype),)