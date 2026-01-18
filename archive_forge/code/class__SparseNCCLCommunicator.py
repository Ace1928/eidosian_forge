import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
class _SparseNCCLCommunicator:

    @classmethod
    def _get_internal_arrays(cls, array):
        if sparse.isspmatrix_coo(array):
            array.sum_duplicates()
            return (array.data, array.row, array.col)
        elif sparse.isspmatrix_csr(array) or sparse.isspmatrix_csc(array):
            return (array.data, array.indptr, array.indices)
        raise TypeError('NCCL is not supported for this type of sparse matrix')

    @classmethod
    def _get_shape_and_sizes(cls, arrays, shape):
        sizes_shape = shape + tuple((a.size for a in arrays))
        return sizes_shape

    @classmethod
    def _exchange_shape_and_sizes(cls, comm, peer, sizes_shape, method, stream):
        if comm._use_mpi:
            if method == 'send':
                sizes_shape = numpy.array(sizes_shape, dtype='q')
                comm._mpi_comm.Send(sizes_shape, dest=peer, tag=1)
                return None
            elif method == 'recv':
                sizes_shape = numpy.empty(5, dtype='q')
                comm._mpi_comm.Recv(sizes_shape, source=peer, tag=1)
                return sizes_shape
            elif method == 'bcast':
                if comm.rank == peer:
                    sizes_shape = numpy.array(sizes_shape, dtype='q')
                else:
                    sizes_shape = numpy.empty(5, dtype='q')
                comm._mpi_comm.Bcast(sizes_shape, root=peer)
                return sizes_shape
            elif method == 'gather':
                sizes_shape = numpy.array(sizes_shape, dtype='q')
                recv_buf = numpy.empty([comm._n_devices, 5], dtype='q')
                comm._mpi_comm.Gather(sizes_shape, recv_buf, peer)
                return recv_buf
            elif method == 'alltoall':
                sizes_shape = numpy.array(sizes_shape, dtype='q')
                recv_buf = numpy.empty([comm._n_devices, 5], dtype='q')
                comm._mpi_comm.Alltoall(sizes_shape, recv_buf)
                return recv_buf
            else:
                raise RuntimeError('Unsupported method')
        else:
            warnings.warn('Using NCCL for transferring sparse arrays metadata. This will cause device synchronization and a huge performance degradation. Please install MPI and `mpi4py` in order to avoid this issue.')
            if method == 'send':
                sizes_shape = cupy.array(sizes_shape, dtype='q')
                cls._send(comm, sizes_shape, peer, sizes_shape.dtype, 5, stream)
                return None
            elif method == 'recv':
                sizes_shape = cupy.empty(5, dtype='q')
                cls._recv(comm, sizes_shape, peer, sizes_shape.dtype, 5, stream)
                return cupy.asnumpy(sizes_shape)
            elif method == 'bcast':
                if comm.rank == peer:
                    sizes_shape = cupy.array(sizes_shape, dtype='q')
                else:
                    sizes_shape = cupy.empty(5, dtype='q')
                _DenseNCCLCommunicator.broadcast(comm, sizes_shape, root=peer, stream=stream)
                return cupy.asnumpy(sizes_shape)
            elif method == 'gather':
                sizes_shape = cupy.array(sizes_shape, dtype='q')
                recv_buf = cupy.empty((comm._n_devices, 5), dtype='q')
                _DenseNCCLCommunicator.gather(comm, sizes_shape, recv_buf, root=peer, stream=stream)
                return cupy.asnumpy(recv_buf)
            elif method == 'alltoall':
                sizes_shape = cupy.array(sizes_shape, dtype='q')
                recv_buf = cupy.empty((comm._n_devices, 5), dtype='q')
                _DenseNCCLCommunicator.all_to_all(comm, sizes_shape, recv_buf, stream=stream)
                return cupy.asnumpy(recv_buf)
            else:
                raise RuntimeError('Unsupported method')

    def _assign_arrays(matrix, arrays, shape):
        if sparse.isspmatrix_coo(matrix):
            matrix.data = arrays[0]
            matrix.row = arrays[1]
            matrix.col = arrays[2]
            matrix._shape = tuple(shape)
        elif sparse.isspmatrix_csr(matrix) or sparse.isspmatrix_csc(matrix):
            matrix.data = arrays[0]
            matrix.indptr = arrays[1]
            matrix.indices = arrays[2]
            matrix._shape = tuple(shape)
        else:
            raise TypeError('NCCL is not supported for this type of sparse matrix')

    @classmethod
    def all_reduce(cls, comm, in_array, out_array, op='sum', stream=None):
        root = 0
        cls.reduce(comm, in_array, out_array, root, op, stream)
        cls.broadcast(comm, out_array, root, stream)

    @classmethod
    def reduce(cls, comm, in_array, out_array, root=0, op='sum', stream=None):
        arrays = cls._get_internal_arrays(in_array)
        shape_and_sizes = cls._get_shape_and_sizes(arrays, in_array.shape)
        shape_and_sizes = cls._exchange_shape_and_sizes(comm, root, shape_and_sizes, 'gather', stream)
        if comm.rank == root:
            if _get_sparse_type(in_array) != _get_sparse_type(out_array):
                raise ValueError('in_array and out_array must be the same format')
            result = in_array
            partial = _make_sparse_empty(in_array.dtype, _get_sparse_type(in_array))
            for peer, ss in enumerate(shape_and_sizes):
                shape = tuple(ss[0:2])
                sizes = ss[2:]
                arrays = [cupy.empty(s, dtype=a.dtype) for s, a in zip(sizes, arrays)]
                if peer != root:
                    nccl.groupStart()
                    for a in arrays:
                        cls._recv(comm, a, peer, a.dtype, a.size, stream)
                    nccl.groupEnd()
                    cls._assign_arrays(partial, arrays, shape)
                    if op == 'sum':
                        result = result + partial
                    elif op == 'prod':
                        result = result * partial
                    else:
                        raise ValueError('Sparse matrix only supports sum/prod reduction')
            cls._assign_arrays(out_array, cls._get_internal_arrays(result), result.shape)
        else:
            nccl.groupStart()
            for a in arrays:
                cls._send(comm, a, root, a.dtype, a.size, stream)
            nccl.groupEnd()

    @classmethod
    def broadcast(cls, comm, in_out_array, root=0, stream=None):
        arrays = cls._get_internal_arrays(in_out_array)
        if comm.rank == root:
            shape_and_sizes = cls._get_shape_and_sizes(arrays, in_out_array.shape)
        else:
            shape_and_sizes = ()
        shape_and_sizes = cls._exchange_shape_and_sizes(comm, root, shape_and_sizes, 'bcast', stream)
        shape = tuple(shape_and_sizes[0:2])
        sizes = shape_and_sizes[2:]
        if comm.rank != root:
            arrays = [cupy.empty(s, dtype=a.dtype) for s, a in zip(sizes, arrays)]
        nccl.groupStart()
        for a in arrays:
            _DenseNCCLCommunicator.broadcast(comm, a, root, stream)
        nccl.groupEnd()
        cls._assign_arrays(in_out_array, arrays, shape)

    @classmethod
    def reduce_scatter(cls, comm, in_array, out_array, count, op='sum', stream=None):
        root = 0
        reduce_out_arrays = []
        if not isinstance(in_array, (list, tuple)):
            raise ValueError('in_array must be a list or a tuple of sparse matrices')
        for s_m in in_array:
            partial_out_array = _make_sparse_empty(s_m.dtype, _get_sparse_type(s_m))
            cls.reduce(comm, s_m, partial_out_array, root, op, stream)
            reduce_out_arrays.append(partial_out_array)
        cls.scatter(comm, reduce_out_arrays, out_array, root, stream)

    @classmethod
    def all_gather(cls, comm, in_array, out_array, count, stream=None):
        root = 0
        gather_out_arrays = []
        cls.gather(comm, in_array, gather_out_arrays, root, stream)
        if comm.rank != root:
            gather_out_arrays = [_make_sparse_empty(in_array.dtype, _get_sparse_type(in_array)) for _ in range(comm._n_devices)]
        for arr in gather_out_arrays:
            cls.broadcast(comm, arr, root, stream)
            out_array.append(arr)

    @classmethod
    def send(cls, comm, array, peer, stream=None):
        arrays = cls._get_internal_arrays(array)
        shape_and_sizes = cls._get_shape_and_sizes(arrays, array.shape)
        cls._exchange_shape_and_sizes(comm, peer, shape_and_sizes, 'send', stream)
        nccl.groupStart()
        for a in arrays:
            cls._send(comm, a, peer, a.dtype, a.size, stream)
        nccl.groupEnd()

    @classmethod
    def _send(cls, comm, array, peer, dtype, count, stream=None):
        dtype = array.dtype.char
        if dtype not in _nccl_dtypes:
            raise TypeError(f'Unknown dtype {array.dtype} for NCCL')
        dtype, count = comm._get_nccl_dtype_and_count(array)
        stream = comm._get_stream(stream)
        comm._comm.send(array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def recv(cls, comm, out_array, peer, stream=None):
        shape_and_sizes = cls._exchange_shape_and_sizes(comm, peer, (), 'recv', stream)
        arrays = cls._get_internal_arrays(out_array)
        shape = tuple(shape_and_sizes[0:2])
        sizes = shape_and_sizes[2:]
        arrs = [cupy.empty(s, dtype=a.dtype) for s, a in zip(sizes, arrays)]
        nccl.groupStart()
        for a in arrs:
            cls._recv(comm, a, peer, a.dtype, a.size, stream)
        nccl.groupEnd()
        cls._assign_arrays(out_array, arrs, shape)

    @classmethod
    def _recv(cls, comm, out_array, peer, dtype, count, stream=None):
        dtype = dtype.char
        if dtype not in _nccl_dtypes:
            raise TypeError(f'Unknown dtype {out_array.dtype} for NCCL')
        dtype, count = comm._get_nccl_dtype_and_count(out_array)
        stream = comm._get_stream(stream)
        comm._comm.recv(out_array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def send_recv(cls, comm, in_array, out_array, peer, stream=None):
        nccl.groupStart()
        cls.send(comm, in_array, peer, stream)
        cls.recv(comm, out_array, peer, stream)
        nccl.groupEnd()

    @classmethod
    def scatter(cls, comm, in_array, out_array, root=0, stream=None):
        if comm.rank == root:
            nccl.groupStart()
            for peer, s_a in enumerate(in_array):
                if peer != root:
                    cls.send(comm, s_a, peer, stream)
            nccl.groupEnd()
            cls._assign_arrays(out_array, cls._get_internal_arrays(in_array[root]), in_array[root].shape)
        else:
            cls.recv(comm, out_array, root, stream)

    @classmethod
    def gather(cls, comm, in_array, out_array, root=0, stream=None):
        if comm.rank == root:
            for peer in range(comm._n_devices):
                res = _make_sparse_empty(in_array.dtype, _get_sparse_type(in_array))
                if peer != root:
                    cls.recv(comm, res, peer, stream)
                else:
                    cls._assign_arrays(res, cls._get_internal_arrays(in_array), in_array.shape)
                out_array.append(res)
        else:
            cls.send(comm, in_array, root, stream)

    @classmethod
    def all_to_all(cls, comm, in_array, out_array, stream=None):
        if len(in_array) != comm._n_devices:
            raise RuntimeError(f'all_to_all requires in_array to have {comm._n_devices}elements, found {len(in_array)}')
        shape_and_sizes = []
        recv_shape_and_sizes = []
        for i, a in enumerate(in_array):
            arrays = cls._get_internal_arrays(a)
            shape_and_sizes.append(cls._get_shape_and_sizes(arrays, a.shape))
        recv_shape_and_sizes = cls._exchange_shape_and_sizes(comm, i, shape_and_sizes, 'alltoall', stream)
        for i in range(comm._n_devices):
            shape = tuple(recv_shape_and_sizes[i][0:2])
            sizes = recv_shape_and_sizes[i][2:]
            s_arrays = cls._get_internal_arrays(in_array[i])
            r_arrays = [cupy.empty(s, dtype=a.dtype) for s, a in zip(sizes, s_arrays)]
            nccl.groupStart()
            for a in s_arrays:
                cls._send(comm, a, i, a.dtype, a.size, stream)
            for a in r_arrays:
                cls._recv(comm, a, i, a.dtype, a.size, stream)
            nccl.groupEnd()
            out_array.append(_make_sparse_empty(in_array[i].dtype, _get_sparse_type(in_array[i])))
            cls._assign_arrays(out_array[i], r_arrays, shape)