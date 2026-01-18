import os
import subprocess as sp
import shlex
import simplejson as json
from traits.trait_errors import TraitError
from ... import config, logging, LooseVersion
from ...utils.provenance import write_provenance
from ...utils.misc import str2bool
from ...utils.filemanip import (
from ...utils.subprocess import run_command
from ...external.due import due
from .traits_extension import traits, isdefined, Undefined
from .specs import (
from .support import (
@classmethod
def _get_filecopy_info(cls):
    """Provides information about file inputs to copy or link to cwd.
        Necessary for pipeline operation
        """
    iflogger.warning('_get_filecopy_info member of Interface was deprecated in nipype-1.1.6 and will be removed in 1.2.0')
    return get_filecopy_info(cls)