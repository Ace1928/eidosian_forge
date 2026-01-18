import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
class MPI4PY:

    def __init__(self, mpi4py_comm=None):
        if mpi4py_comm is None:
            from mpi4py import MPI
            mpi4py_comm = MPI.COMM_WORLD
        self.comm = mpi4py_comm

    @property
    def rank(self):
        return self.comm.rank

    @property
    def size(self):
        return self.comm.size

    def _returnval(self, a, b):
        """Behave correctly when working on scalars/arrays.

        Either input is an array and we in-place write b (output from
        mpi4py) back into a, or input is a scalar and we return the
        corresponding output scalar."""
        if np.isscalar(a):
            assert np.isscalar(b)
            return b
        else:
            assert not np.isscalar(b)
            a[:] = b
            return None

    def sum(self, a, root=-1):
        if root == -1:
            b = self.comm.allreduce(a)
        else:
            b = self.comm.reduce(a, root)
        return self._returnval(a, b)

    def split(self, split_size=None):
        """Divide the communicator."""
        if not split_size:
            split_size = self.size
        color = int(self.rank // (self.size / split_size))
        key = int(self.rank % (self.size / split_size))
        comm = self.comm.Split(color, key)
        return MPI4PY(comm)

    def barrier(self):
        self.comm.barrier()

    def abort(self, code):
        self.comm.Abort(code)

    def broadcast(self, a, root):
        b = self.comm.bcast(a, root=root)
        if self.rank == root:
            if np.isscalar(a):
                return a
            return
        return self._returnval(a, b)