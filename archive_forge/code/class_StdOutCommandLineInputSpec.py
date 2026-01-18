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
class StdOutCommandLineInputSpec(CommandLineInputSpec):
    out_file = File(argstr='> %s', position=-1, genfile=True)