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
def _pre_run_hook(self, runtime):
    """
        Perform any pre-_run_interface() processing

        Subclasses may override this function to modify ``runtime`` object or
        interface state

        MUST return runtime object
        """
    return runtime