from enum import Enum
from dataclasses import dataclass
from datetime import timedelta
class Backend(object):
    """A class to represent different backends."""
    NCCL = 'nccl'
    MPI = 'mpi'
    GLOO = 'gloo'
    UNRECOGNIZED = 'unrecognized'

    def __new__(cls, name: str):
        backend = getattr(Backend, name.upper(), Backend.UNRECOGNIZED)
        if backend == Backend.UNRECOGNIZED:
            raise ValueError("Unrecognized backend: '{}'. Only NCCL is supported".format(name))
        if backend == Backend.MPI:
            raise RuntimeError('Ray does not support MPI backend.')
        return backend