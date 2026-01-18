import os
from packaging.version import Version
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, Undefined, PackageInfo
from ...utils.filemanip import split_filename
def _omp_update(self):
    if self.inputs.omp_core_val:
        self.inputs.environ['OMP_NUM_THREADS'] = str(self.inputs.omp_core_val)
        self.num_threads = self.inputs.omp_core_val
    else:
        if 'OMP_NUM_THREADS' in self.inputs.environ:
            del self.inputs.environ['OMP_NUM_THREADS']
        self.num_threads = 1