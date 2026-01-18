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
def _check_mandatory_inputs(self):
    """Raises an exception if a mandatory input is Undefined"""
    for name, spec in list(self.inputs.traits(mandatory=True).items()):
        value = getattr(self.inputs, name)
        self._check_xor(spec, name, value)
        if not isdefined(value) and spec.xor is None:
            msg = "%s requires a value for input '%s'. For a list of required inputs, see %s.help()" % (self.__class__.__name__, name, self.__class__.__name__)
            raise ValueError(msg)
        if isdefined(value):
            self._check_requires(spec, name, value)
    for name, spec in list(self.inputs.traits(mandatory=None, transient=None).items()):
        self._check_requires(spec, name, getattr(self.inputs, name))