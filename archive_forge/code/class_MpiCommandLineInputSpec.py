import os
from inspect import isclass
from copy import deepcopy
from warnings import warn
from packaging.version import Version
from traits.trait_errors import TraitError
from traits.trait_handlers import TraitDictObject, TraitListObject
from ...utils.filemanip import md5, hash_infile, hash_timestamp
from .traits_extension import (
from ... import config, __version__
class MpiCommandLineInputSpec(CommandLineInputSpec):
    use_mpi = traits.Bool(False, desc='Whether or not to run the command with mpiexec', usedefault=True)
    n_procs = traits.Int(desc='Num processors to specify to mpiexec. Do not specify if this is managed externally (e.g. through SGE)')