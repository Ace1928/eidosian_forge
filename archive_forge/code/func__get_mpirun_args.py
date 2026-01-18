import argparse
import os
import subprocess
import sys
import warnings
from ast import literal_eval
from shutil import which
from typing import Any, Dict, List, Tuple
import torch
from ..commands.config.config_args import SageMakerConfig
from ..utils import (
from ..utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from ..utils.other import is_port_in_use, merge_dicts
from .dataclasses import DistributedType, SageMakerDistributedType
def _get_mpirun_args():
    """
    Determines the executable and argument names for mpirun, based on the type of install. The supported MPI programs
    are: OpenMPI, Intel MPI, or MVAPICH.

    Returns: Program name and arg names for hostfile, num processes, and processes per node
    """
    mpi_apps = [x for x in ['mpirun', 'mpiexec'] if which(x)]
    if len(mpi_apps) == 0:
        raise OSError('mpirun or mpiexec were not found. Ensure that Intel MPI, Open MPI, or MVAPICH are installed.')
    mpi_app = mpi_apps[0]
    mpirun_version = subprocess.check_output([mpi_app, '--version'])
    if b'Open MPI' in mpirun_version:
        return (mpi_app, '--hostfile', '-n', '--npernode')
    else:
        return (mpi_app, '-f', '-n', '-ppn')