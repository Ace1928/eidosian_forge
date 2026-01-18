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
class StdOutCommandLine(CommandLine):
    input_spec = StdOutCommandLineInputSpec

    def _gen_filename(self, name):
        return self._gen_outfilename() if name == 'out_file' else None

    def _gen_outfilename(self):
        raise NotImplementedError