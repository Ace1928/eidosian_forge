import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class ApplyTransform(SPMCommand):
    """Uses SPM to apply transform stored in a .mat file to given file

    Examples
    --------

    >>> import nipype.interfaces.spm.utils as spmu
    >>> applymat = spmu.ApplyTransform()
    >>> applymat.inputs.in_file = 'functional.nii'
    >>> applymat.inputs.mat = 'func_to_struct.mat'
    >>> applymat.run() # doctest: +SKIP

    """
    input_spec = ApplyTransformInputSpec
    output_spec = ApplyTransformOutputSpec

    def _make_matlab_command(self, _):
        """checks for SPM, generates script"""
        outputs = self._list_outputs()
        self.inputs.out_file = outputs['out_file']
        script = "\n        infile = '%s';\n        outfile = '%s'\n        transform = load('%s');\n\n        V = spm_vol(infile);\n        X = spm_read_vols(V);\n        [p n e v] = spm_fileparts(V.fname);\n        V.mat = transform.M * V.mat;\n        V.fname = fullfile(outfile);\n        spm_write_vol(V,X);\n\n        " % (self.inputs.in_file, self.inputs.out_file, self.inputs.mat)
        return script

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = os.path.abspath(self._gen_outfilename())
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_trans.nii'