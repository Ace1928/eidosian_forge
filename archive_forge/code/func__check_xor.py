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
def _check_xor(self, spec, name, value):
    """check if mutually exclusive inputs are satisfied"""
    if spec.xor:
        values = [isdefined(getattr(self.inputs, field)) for field in spec.xor]
        if not any(values) and (not isdefined(value)):
            msg = "%s requires a value for one of the inputs '%s'. For a list of required inputs, see %s.help()" % (self.__class__.__name__, ', '.join(spec.xor), self.__class__.__name__)
            raise ValueError(msg)