import pytest
import os
import stat
import pickle
import tempfile
import subprocess
import sys
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
from h5py._hl.files import direct_vfd
from h5py import File
import h5py
from .. import h5
import pathlib
import sys
import h5py
@pytest.mark.mpi
class TestMPI:

    def test_mpio(self, mpi_file_name):
        """ MPIO driver and options """
        from mpi4py import MPI
        with File(mpi_file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
            assert f
            assert f.driver == 'mpio'

    def test_mpio_append(self, mpi_file_name):
        """ Testing creation of file with append """
        from mpi4py import MPI
        with File(mpi_file_name, 'a', driver='mpio', comm=MPI.COMM_WORLD) as f:
            assert f
            assert f.driver == 'mpio'

    def test_mpi_atomic(self, mpi_file_name):
        """ Enable atomic mode for MPIO driver """
        from mpi4py import MPI
        with File(mpi_file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
            assert not f.atomic
            f.atomic = True
            assert f.atomic

    def test_close_multiple_mpio_driver(self, mpi_file_name):
        """ MPIO driver and options """
        from mpi4py import MPI
        f = File(mpi_file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        f.create_group('test')
        f.close()
        f.close()