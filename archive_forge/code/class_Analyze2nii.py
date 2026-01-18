import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class Analyze2nii(SPMCommand):
    input_spec = Analyze2niiInputSpec
    output_spec = Analyze2niiOutputSpec

    def _make_matlab_command(self, _):
        script = "V = spm_vol('%s');\n" % self.inputs.analyze_file
        _, name, _ = split_filename(self.inputs.analyze_file)
        self.output_name = os.path.join(os.getcwd(), name + '.nii')
        script += '[Y, XYZ] = spm_read_vols(V);\n'
        script += "V.fname = '%s';\n" % self.output_name
        script += 'spm_write_vol(V, Y);\n'
        return script

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['nifti_file'] = self.output_name
        return outputs