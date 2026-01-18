import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
class _DenseNCCLCommunicator:

    @classmethod
    def all_reduce(cls, comm, in_array, out_array, op='sum', stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array)
        op = comm._get_op(op, in_array.dtype.char)
        comm._comm.allReduce(in_array.data.ptr, out_array.data.ptr, count, dtype, op, stream)

    @classmethod
    def reduce(cls, comm, in_array, out_array, root=0, op='sum', stream=None):
        comm._check_contiguous(in_array)
        if comm.rank == root:
            comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array)
        op = comm._get_op(op, in_array.dtype.char)
        comm._comm.reduce(in_array.data.ptr, out_array.data.ptr, count, dtype, op, root, stream)

    @classmethod
    def broadcast(cls, comm, in_out_array, root=0, stream=None):
        comm._check_contiguous(in_out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_out_array)
        comm._comm.broadcast(in_out_array.data.ptr, in_out_array.data.ptr, count, dtype, root, stream)

    @classmethod
    def reduce_scatter(cls, comm, in_array, out_array, count, op='sum', stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array, count)
        op = comm._get_op(op, in_array.dtype.char)
        comm._comm.reduceScatter(in_array.data.ptr, out_array.data.ptr, count, dtype, op, stream)

    @classmethod
    def all_gather(cls, comm, in_array, out_array, count, stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array, count)
        comm._comm.allGather(in_array.data.ptr, out_array.data.ptr, count, dtype, stream)

    @classmethod
    def send(cls, comm, array, peer, stream=None):
        comm._check_contiguous(array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(array)
        cls._send(comm, array, peer, dtype, count, stream)

    @classmethod
    def _send(cls, comm, array, peer, dtype, count, stream=None):
        comm._comm.send(array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def recv(cls, comm, out_array, peer, stream=None):
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(out_array)
        cls._recv(comm, out_array, peer, dtype, count, stream)

    @classmethod
    def _recv(cls, comm, out_array, peer, dtype, count, stream=None):
        comm._comm.recv(out_array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def send_recv(cls, comm, in_array, out_array, peer, stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        idtype, icount = comm._get_nccl_dtype_and_count(in_array)
        odtype, ocount = comm._get_nccl_dtype_and_count(out_array)
        nccl.groupStart()
        cls._send(comm, in_array, peer, idtype, icount, stream)
        cls._recv(comm, out_array, peer, odtype, ocount, stream)
        nccl.groupEnd()

    @classmethod
    def scatter(cls, comm, in_array, out_array, root=0, stream=None):
        if in_array.shape[0] != comm._n_devices:
            raise RuntimeError(f'scatter requires in_array to have {comm._n_devices}elements in its first dimension, found {in_array.shape}')
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        nccl.groupStart()
        if root == comm.rank:
            for i in range(comm._n_devices):
                array = in_array[i]
                idtype, icount = comm._get_nccl_dtype_and_count(array)
                cls._send(comm, array, i, idtype, icount, stream)
        dtype, count = comm._get_nccl_dtype_and_count(out_array)
        cls._recv(comm, out_array, root, dtype, count, stream)
        nccl.groupEnd()

    @classmethod
    def gather(cls, comm, in_array, out_array, root=0, stream=None):
        if out_array.shape[0] != comm._n_devices:
            raise RuntimeError(f'gather requires out_array to have {comm._n_devices}elements in its first dimension, found {out_array.shape}')
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        nccl.groupStart()
        if root == comm.rank:
            for i in range(comm._n_devices):
                array = out_array[i]
                odtype, ocount = comm._get_nccl_dtype_and_count(array)
                cls._recv(comm, array, i, odtype, ocount, stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array)
        cls._send(comm, in_array, root, dtype, count, stream)
        nccl.groupEnd()

    @classmethod
    def all_to_all(cls, comm, in_array, out_array, stream=None):
        if out_array.shape[0] != comm._n_devices:
            raise RuntimeError(f'all_to_all requires in_array to have {comm._n_devices}elements in its first dimension, found {in_array.shape}')
        if out_array.shape[0] != comm._n_devices:
            raise RuntimeError(f'all_to_all requires out_array to have {comm._n_devices}elements in its first dimension, found {out_array.shape}')
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        idtype, icount = comm._get_nccl_dtype_and_count(in_array[0])
        odtype, ocount = comm._get_nccl_dtype_and_count(out_array[0])
        nccl.groupStart()
        for i in range(comm._n_devices):
            cls._send(comm, in_array[i], i, idtype, icount, stream)
            cls._recv(comm, out_array[i], i, odtype, ocount, stream)
        nccl.groupEnd()