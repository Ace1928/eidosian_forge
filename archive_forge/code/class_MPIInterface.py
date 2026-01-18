from collections import OrderedDict
import importlib
class MPIInterface:
    __have_mpi__ = None

    def __init__(self):
        if MPIInterface.__have_mpi__ is None:
            try:
                globals()['MPI'] = importlib.import_module('mpi4py.MPI')
                MPIInterface.__have_mpi__ = True
            except:
                MPIInterface.__have_mpi__ = False
        self._comm = None
        self._size = None
        self._rank = None
        if self.have_mpi:
            self._comm = MPI.COMM_WORLD
            self._size = self._comm.Get_size()
            self._rank = self._comm.Get_rank()

    @property
    def have_mpi(self):
        assert MPIInterface.__have_mpi__ is not None
        return MPIInterface.__have_mpi__

    @property
    def comm(self):
        return self._comm

    @property
    def rank(self):
        return self._rank

    @property
    def size(self):
        return self._size