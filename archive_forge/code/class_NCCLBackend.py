import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
class NCCLBackend(_Backend):
    """Interface that uses NVIDIA's NCCL to perform communications.

    Args:
        n_devices (int): Total number of devices that will be used in the
            distributed execution.
        rank (int): Unique id of the GPU that the communicator is associated to
            its value needs to be `0 <= rank < n_devices`.
        host (str, optional): host address for the process rendezvous on
            initialization. Defaults to `"127.0.0.1"`.
        port (int, optional): port used for the process rendezvous on
            initialization. Defaults to `13333`.
        use_mpi(bool, optional): switch between MPI and use the included TCP
            server for initialization & synchronization. Defaults to `False`.
    """

    def __init__(self, n_devices, rank, host=_store._DEFAULT_HOST, port=_store._DEFAULT_PORT, use_mpi=False):
        super().__init__(n_devices, rank, host, port)
        self._use_mpi = _mpi_available and use_mpi
        if self._use_mpi:
            self._init_with_mpi(n_devices, rank)
        else:
            self._init_with_tcp_store(n_devices, rank, host, port)

    def _init_with_mpi(self, n_devices, rank):
        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_rank = self._mpi_comm.Get_rank()
        self._mpi_comm.Barrier()
        nccl_id = None
        if self._mpi_rank == 0:
            nccl_id = nccl.get_unique_id()
        nccl_id = self._mpi_comm.bcast(nccl_id, root=0)
        self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)

    def _init_with_tcp_store(self, n_devices, rank, host, port):
        nccl_id = None
        if rank == 0:
            self._store.run(host, port)
            nccl_id = nccl.get_unique_id()
            shifted_nccl_id = bytes([b + 128 for b in nccl_id])
            self._store_proxy['nccl_id'] = shifted_nccl_id
            self._store_proxy.barrier()
        else:
            self._store_proxy.barrier()
            nccl_id = self._store_proxy['nccl_id']
            nccl_id = tuple([int(b) - 128 for b in nccl_id])
        self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)

    def _check_contiguous(self, array):
        if not array.flags.c_contiguous and (not array.flags.f_contiguous):
            raise RuntimeError('NCCL requires arrays to be either c- or f-contiguous')

    def _get_nccl_dtype_and_count(self, array, count=None):
        dtype = array.dtype.char
        if dtype not in _nccl_dtypes:
            raise TypeError(f'Unknown dtype {array.dtype} for NCCL')
        nccl_dtype = _nccl_dtypes[dtype]
        if count is None:
            count = array.size
        if dtype in 'FD':
            return (nccl_dtype, 2 * count)
        return (nccl_dtype, count)

    def _get_stream(self, stream):
        if stream is None:
            stream = cupy.cuda.stream.get_current_stream()
        return stream.ptr

    def _get_op(self, op, dtype):
        if op not in _nccl_ops:
            raise RuntimeError(f'Unknown op {op} for NCCL')
        if dtype in 'FD' and op != nccl.NCCL_SUM:
            raise ValueError('Only nccl.SUM is supported for complex arrays')
        return _nccl_ops[op]

    def _dispatch_arg_type(self, function, args):
        comm_class = _DenseNCCLCommunicator
        if isinstance(args[0], (list, tuple)) and sparse.issparse(args[0][0]) or sparse.issparse(args[0]):
            comm_class = _SparseNCCLCommunicator
        getattr(comm_class, function)(self, *args)

    def all_reduce(self, in_array, out_array, op='sum', stream=None):
        """Performs an all reduce operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('all_reduce', (in_array, out_array, op, stream))

    def reduce(self, in_array, out_array, root=0, op='sum', stream=None):
        """Performs a reduce operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
                will only be modified by the `root` process.
            root (int, optional): rank of the process that will perform the
                reduction. Defaults to `0`.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('reduce', (in_array, out_array, root, op, stream))

    def broadcast(self, in_out_array, root=0, stream=None):
        """Performs a broadcast operation.

        Args:
            in_out_array (cupy.ndarray): array to be sent for `root` rank.
                Other ranks will receive the broadcast data here.
            root (int, optional): rank of the process that will send the
                broadcast. Defaults to `0`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('broadcast', (in_out_array, root, stream))

    def reduce_scatter(self, in_array, out_array, count, op='sum', stream=None):
        """Performs a reduce scatter operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            count (int): Number of elements to send to each rank.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('reduce_scatter', (in_array, out_array, count, op, stream))

    def all_gather(self, in_array, out_array, count, stream=None):
        """Performs an all gather operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            count (int): Number of elements to send to each rank.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('all_gather', (in_array, out_array, count, stream))

    def send(self, array, peer, stream=None):
        """Performs a send operation.

        Args:
            array (cupy.ndarray): array to be sent.
            peer (int): rank of the process `array` will be sent to.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('send', (array, peer, stream))

    def recv(self, out_array, peer, stream=None):
        """Performs a receive operation.

        Args:
            array (cupy.ndarray): array used to receive data.
            peer (int): rank of the process `array` will be received from.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('recv', (out_array, peer, stream))

    def send_recv(self, in_array, out_array, peer, stream=None):
        """Performs a send and receive operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array used to receive data.
            peer (int): rank of the process to send `in_array` and receive
                `out_array`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('send_recv', (in_array, out_array, peer, stream))

    def scatter(self, in_array, out_array, root=0, stream=None):
        """Performs a scatter operation.

        Args:
            in_array (cupy.ndarray): array to be sent. Its shape must be
                `(total_ranks, ...)`.
            out_array (cupy.ndarray): array where the result with be stored.
            root (int): rank that will send the `in_array` to other ranks.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('scatter', (in_array, out_array, root, stream))

    def gather(self, in_array, out_array, root=0, stream=None):
        """Performs a gather operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
                Its shape must be `(total_ranks, ...)`.
            root (int): rank that will receive `in_array` from other ranks.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('gather', (in_array, out_array, root, stream))

    def all_to_all(self, in_array, out_array, stream=None):
        """Performs an all to all operation.

        Args:
            in_array (cupy.ndarray): array to be sent. Its shape must be
                `(total_ranks, ...)`.
            out_array (cupy.ndarray): array where the result with be stored.
                Its shape must be `(total_ranks, ...)`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('all_to_all', (in_array, out_array, stream))

    def barrier(self):
        """Performs a barrier operation.

        The barrier is done in the cpu and is a explicit synchronization
        mechanism that halts the thread progression.
        """
        if self._use_mpi:
            self._mpi_comm.Barrier()
        else:
            self._store_proxy.barrier()