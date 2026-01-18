import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Allineate(AFNICommand):
    """Program to align one dataset (the 'source') to a base dataset

    For complete details, see the `3dAllineate Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAllineate.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> allineate = afni.Allineate()
    >>> allineate.inputs.in_file = 'functional.nii'
    >>> allineate.inputs.out_file = 'functional_allineate.nii'
    >>> allineate.inputs.in_matrix = 'cmatrix.mat'
    >>> allineate.cmdline
    '3dAllineate -source functional.nii -prefix functional_allineate.nii -1Dmatrix_apply cmatrix.mat'
    >>> res = allineate.run()  # doctest: +SKIP

    >>> allineate = afni.Allineate()
    >>> allineate.inputs.in_file = 'functional.nii'
    >>> allineate.inputs.reference = 'structural.nii'
    >>> allineate.inputs.allcostx = 'out.allcostX.txt'
    >>> allineate.cmdline
    '3dAllineate -source functional.nii -base structural.nii -allcostx |& tee out.allcostX.txt'
    >>> res = allineate.run()  # doctest: +SKIP

    >>> allineate = afni.Allineate()
    >>> allineate.inputs.in_file = 'functional.nii'
    >>> allineate.inputs.reference = 'structural.nii'
    >>> allineate.inputs.nwarp_fixmot = ['X', 'Y']
    >>> allineate.cmdline
    '3dAllineate -source functional.nii -nwarp_fixmotX -nwarp_fixmotY -prefix functional_allineate -base structural.nii'
    >>> res = allineate.run()  # doctest: +SKIP
    """
    _cmd = '3dAllineate'
    input_spec = AllineateInputSpec
    output_spec = AllineateOutputSpec

    def _list_outputs(self):
        outputs = super(Allineate, self)._list_outputs()
        if self.inputs.out_weight_file:
            outputs['out_weight_file'] = op.abspath(self.inputs.out_weight_file)
        if self.inputs.out_matrix:
            ext = split_filename(self.inputs.out_matrix)[-1]
            if ext.lower() not in ['.1d', '.1D']:
                outputs['out_matrix'] = self._gen_fname(self.inputs.out_matrix, suffix='.aff12.1D')
            else:
                outputs['out_matrix'] = op.abspath(self.inputs.out_matrix)
        if self.inputs.out_param_file:
            ext = split_filename(self.inputs.out_param_file)[-1]
            if ext.lower() not in ['.1d', '.1D']:
                outputs['out_param_file'] = self._gen_fname(self.inputs.out_param_file, suffix='.param.1D')
            else:
                outputs['out_param_file'] = op.abspath(self.inputs.out_param_file)
        if self.inputs.allcostx:
            outputs['allcostX'] = os.path.abspath(self.inputs.allcostx)
        return outputs