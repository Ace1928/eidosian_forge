import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TCorr1D(AFNICommand):
    """Computes the correlation coefficient between each voxel time series
    in the input 3D+time dataset.

    For complete details, see the `3dTcorr1D Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTcorr1D.html>`_

    >>> from nipype.interfaces import afni
    >>> tcorr1D = afni.TCorr1D()
    >>> tcorr1D.inputs.xset= 'u_rc1s1_Template.nii'
    >>> tcorr1D.inputs.y_1d = 'seed.1D'
    >>> tcorr1D.cmdline
    '3dTcorr1D -prefix u_rc1s1_Template_correlation.nii.gz  u_rc1s1_Template.nii  seed.1D'
    >>> res = tcorr1D.run()  # doctest: +SKIP

    """
    _cmd = '3dTcorr1D'
    input_spec = TCorr1DInputSpec
    output_spec = TCorr1DOutputSpec