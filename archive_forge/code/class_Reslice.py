import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class Reslice(SPMCommand):
    """uses  spm_reslice to resample in_file into space of space_defining"""
    input_spec = ResliceInputSpec
    output_spec = ResliceOutputSpec

    def _make_matlab_command(self, _):
        """generates script"""
        if not isdefined(self.inputs.out_file):
            self.inputs.out_file = fname_presuffix(self.inputs.in_file, prefix='r')
        script = "\n        flags.mean = 0;\n        flags.which = 1;\n        flags.mask = 0;\n        flags.interp = %d;\n        infiles = strvcat('%s', '%s');\n        invols = spm_vol(infiles);\n        spm_reslice(invols, flags);\n        " % (self.inputs.interp, self.inputs.space_defining, self.inputs.in_file)
        return script

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs