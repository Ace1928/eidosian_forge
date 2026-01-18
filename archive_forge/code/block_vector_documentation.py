import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector

        Creates a parallel MPIBlockVector from this BlockVector

        Parameters
        ----------
        rank_ownership: array_like
            Array_like of size nblocks. Each entry defines ownership of each block.
            There are two types of ownership. Block that are owned by all processor,
            and blocks owned by a single processor. If a block is owned by all
            processors then its ownership is -1. Otherwise, if a block is owned by
            a single processor, then its ownership is equal to the rank of the
            processor.
        mpi_comm: MPI communicator
            An MPI communicator. Tyically MPI.COMM_WORLD

        