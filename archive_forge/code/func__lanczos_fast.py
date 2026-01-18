import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def _lanczos_fast(A, n, ncv):
    cublas_handle = device.get_cublas_handle()
    cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
    if A.dtype.char == 'f':
        dotc = _cublas.sdot
        nrm2 = _cublas.snrm2
        gemv = _cublas.sgemv
        axpy = _cublas.saxpy
    elif A.dtype.char == 'd':
        dotc = _cublas.ddot
        nrm2 = _cublas.dnrm2
        gemv = _cublas.dgemv
        axpy = _cublas.daxpy
    elif A.dtype.char == 'F':
        dotc = _cublas.cdotc
        nrm2 = _cublas.scnrm2
        gemv = _cublas.cgemv
        axpy = _cublas.caxpy
    elif A.dtype.char == 'D':
        dotc = _cublas.zdotc
        nrm2 = _cublas.dznrm2
        gemv = _cublas.zgemv
        axpy = _cublas.zaxpy
    else:
        raise TypeError('invalid dtype ({})'.format(A.dtype))
    cusparse_handle = None
    if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        cusparse_handle = device.get_cusparse_handle()
        spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        spmv_alpha = numpy.array(1.0, A.dtype)
        spmv_beta = numpy.array(0.0, A.dtype)
        spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
    v = cupy.empty((n,), dtype=A.dtype)
    uu = cupy.empty((ncv,), dtype=A.dtype)
    vv = cupy.empty((n,), dtype=A.dtype)
    b = cupy.empty((), dtype=A.dtype)
    one = numpy.array(1.0, dtype=A.dtype)
    zero = numpy.array(0.0, dtype=A.dtype)
    mone = numpy.array(-1.0, dtype=A.dtype)
    outer_A = A

    def aux(A, V, u, alpha, beta, i_start, i_end):
        assert A is outer_A
        if cusparse_handle is not None:
            spmv_desc_A = cusparse.SpMatDescriptor.create(A)
            spmv_desc_v = cusparse.DnVecDescriptor.create(v)
            spmv_desc_u = cusparse.DnVecDescriptor.create(u)
            buff_size = _cusparse.spMV_bufferSize(cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data, spmv_desc_A.desc, spmv_desc_v.desc, spmv_beta.ctypes.data, spmv_desc_u.desc, spmv_cuda_dtype, spmv_alg)
            spmv_buff = cupy.empty(buff_size, cupy.int8)
        v[...] = V[i_start]
        for i in range(i_start, i_end):
            if cusparse_handle is None:
                u[...] = A @ v
            else:
                _cusparse.spMV(cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data, spmv_desc_A.desc, spmv_desc_v.desc, spmv_beta.ctypes.data, spmv_desc_u.desc, spmv_cuda_dtype, spmv_alg, spmv_buff.data.ptr)
            _cublas.setPointerMode(cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                dotc(cublas_handle, n, v.data.ptr, 1, u.data.ptr, 1, alpha.data.ptr + i * alpha.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            vv.fill(0)
            b[...] = beta[i - 1]
            _cublas.setPointerMode(cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                axpy(cublas_handle, n, alpha.data.ptr + i * alpha.itemsize, v.data.ptr, 1, vv.data.ptr, 1)
                axpy(cublas_handle, n, b.data.ptr, V[i - 1].data.ptr, 1, vv.data.ptr, 1)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            axpy(cublas_handle, n, mone.ctypes.data, vv.data.ptr, 1, u.data.ptr, 1)
            gemv(cublas_handle, _cublas.CUBLAS_OP_C, n, i + 1, one.ctypes.data, V.data.ptr, n, u.data.ptr, 1, zero.ctypes.data, uu.data.ptr, 1)
            gemv(cublas_handle, _cublas.CUBLAS_OP_N, n, i + 1, mone.ctypes.data, V.data.ptr, n, uu.data.ptr, 1, one.ctypes.data, u.data.ptr, 1)
            alpha[i] += uu[i]
            _cublas.setPointerMode(cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                nrm2(cublas_handle, n, u.data.ptr, 1, beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            if i >= i_end - 1:
                break
            _kernel_normalize(u, beta, i, n, v, V)
    return aux