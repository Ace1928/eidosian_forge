import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Detrend(AFNICommand):
    """This program removes components from voxel time series using
    linear least squares

    For complete details, see the `3dDetrend Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDetrend.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> detrend = afni.Detrend()
    >>> detrend.inputs.in_file = 'functional.nii'
    >>> detrend.inputs.args = '-polort 2'
    >>> detrend.inputs.outputtype = 'AFNI'
    >>> detrend.cmdline
    '3dDetrend -polort 2 -prefix functional_detrend functional.nii'
    >>> res = detrend.run()  # doctest: +SKIP

    """
    _cmd = '3dDetrend'
    input_spec = DetrendInputSpec
    output_spec = AFNICommandOutputSpec