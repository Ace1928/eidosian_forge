import os
from packaging.version import Version
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, Undefined, PackageInfo
from ...utils.filemanip import split_filename
class NiftyRegCommandInputSpec(CommandLineInputSpec):
    """Input Spec for niftyreg interfaces."""
    omp_core_val = traits.Int(int(os.environ.get('OMP_NUM_THREADS', '1')), desc='Number of openmp thread to use', argstr='-omp %i', usedefault=True)