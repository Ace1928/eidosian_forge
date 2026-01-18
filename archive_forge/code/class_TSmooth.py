import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TSmooth(AFNICommand):
    """Smooths each voxel time series in a 3D+time dataset and produces
    as output a new 3D+time dataset (e.g., lowpass filter in time).

    For complete details, see the `3dTsmooth Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTSmooth.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> from nipype.testing import  example_data
    >>> smooth = afni.TSmooth()
    >>> smooth.inputs.in_file = 'functional.nii'
    >>> smooth.inputs.adaptive = 5
    >>> smooth.cmdline
    '3dTsmooth -adaptive 5 -prefix functional_smooth functional.nii'
    >>> res = smooth.run()  # doctest: +SKIP

    """
    _cmd = '3dTsmooth'
    input_spec = TSmoothInputSpec
    output_spec = AFNICommandOutputSpec