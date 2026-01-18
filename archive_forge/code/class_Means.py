import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Means(AFNICommand):
    """Takes the voxel-by-voxel mean of all input datasets using 3dMean

    For complete details, see the `3dMean Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dMean.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> means = afni.Means()
    >>> means.inputs.in_file_a = 'im1.nii'
    >>> means.inputs.in_file_b = 'im2.nii'
    >>> means.inputs.out_file =  'output.nii'
    >>> means.cmdline
    '3dMean -prefix output.nii im1.nii im2.nii'
    >>> res = means.run()  # doctest: +SKIP

    >>> from nipype.interfaces import afni
    >>> means = afni.Means()
    >>> means.inputs.in_file_a = 'im1.nii'
    >>> means.inputs.out_file =  'output.nii'
    >>> means.inputs.datum = 'short'
    >>> means.cmdline
    '3dMean -datum short -prefix output.nii im1.nii'
    >>> res = means.run()  # doctest: +SKIP

    """
    _cmd = '3dMean'
    input_spec = MeansInputSpec
    output_spec = AFNICommandOutputSpec