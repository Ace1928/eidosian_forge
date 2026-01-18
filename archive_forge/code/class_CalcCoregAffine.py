import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class CalcCoregAffine(SPMCommand):
    """Uses SPM (spm_coreg) to calculate the transform mapping
    moving to target. Saves Transform in mat (matlab binary file)
    Also saves inverse transform

    Examples
    --------

    >>> import nipype.interfaces.spm.utils as spmu
    >>> coreg = spmu.CalcCoregAffine(matlab_cmd='matlab-spm8')
    >>> coreg.inputs.target = 'structural.nii'
    >>> coreg.inputs.moving = 'functional.nii'
    >>> coreg.inputs.mat = 'func_to_struct.mat'
    >>> coreg.run() # doctest: +SKIP

    .. note::

     * the output file mat is saves as a matlab binary file
     * calculating the transforms does NOT change either input image
       it does not **move** the moving image, only calculates the transform
       that can be used to move it
    """
    input_spec = CalcCoregAffineInputSpec
    output_spec = CalcCoregAffineOutputSpec

    def _make_inv_file(self):
        """makes filename to hold inverse transform if not specified"""
        invmat = fname_presuffix(self.inputs.mat, prefix='inverse_')
        return invmat

    def _make_mat_file(self):
        """makes name for matfile if doesn exist"""
        pth, mv, _ = split_filename(self.inputs.moving)
        _, tgt, _ = split_filename(self.inputs.target)
        mat = os.path.join(pth, '%s_to_%s.mat' % (mv, tgt))
        return mat

    def _make_matlab_command(self, _):
        """checks for SPM, generates script"""
        if not isdefined(self.inputs.mat):
            self.inputs.mat = self._make_mat_file()
        if not isdefined(self.inputs.invmat):
            self.inputs.invmat = self._make_inv_file()
        script = "\n        target = '%s';\n        moving = '%s';\n        targetv = spm_vol(target);\n        movingv = spm_vol(moving);\n        x = spm_coreg(targetv, movingv);\n        M = spm_matrix(x);\n        save('%s' , 'M' );\n        M = inv(M);\n        save('%s','M')\n        " % (self.inputs.target, self.inputs.moving, self.inputs.mat, self.inputs.invmat)
        return script

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['mat'] = os.path.abspath(self.inputs.mat)
        outputs['invmat'] = os.path.abspath(self.inputs.invmat)
        return outputs