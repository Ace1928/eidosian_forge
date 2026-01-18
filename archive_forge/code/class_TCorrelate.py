import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TCorrelate(AFNICommand):
    """Computes the correlation coefficient between corresponding voxel
    time series in two input 3D+time datasets 'xset' and 'yset'

    For complete details, see the `3dTcorrelate Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTcorrelate.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> tcorrelate = afni.TCorrelate()
    >>> tcorrelate.inputs.xset= 'u_rc1s1_Template.nii'
    >>> tcorrelate.inputs.yset = 'u_rc1s2_Template.nii'
    >>> tcorrelate.inputs.out_file = 'functional_tcorrelate.nii.gz'
    >>> tcorrelate.inputs.polort = -1
    >>> tcorrelate.inputs.pearson = True
    >>> tcorrelate.cmdline
    '3dTcorrelate -prefix functional_tcorrelate.nii.gz -pearson -polort -1 u_rc1s1_Template.nii u_rc1s2_Template.nii'
    >>> res = tcarrelate.run()  # doctest: +SKIP

    """
    _cmd = '3dTcorrelate'
    input_spec = TCorrelateInputSpec
    output_spec = AFNICommandOutputSpec