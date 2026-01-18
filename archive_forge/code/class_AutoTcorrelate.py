import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class AutoTcorrelate(AFNICommand):
    """Computes the correlation coefficient between the time series of each
    pair of voxels in the input dataset, and stores the output into a
    new anatomical bucket dataset [scaled to shorts to save memory space].

    For complete details, see the `3dAutoTcorrelate Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAutoTcorrelate.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> corr = afni.AutoTcorrelate()
    >>> corr.inputs.in_file = 'functional.nii'
    >>> corr.inputs.polort = -1
    >>> corr.inputs.eta2 = True
    >>> corr.inputs.mask = 'mask.nii'
    >>> corr.inputs.mask_only_targets = True
    >>> corr.cmdline  # doctest: +ELLIPSIS
    '3dAutoTcorrelate -eta2 -mask mask.nii -mask_only_targets -prefix functional_similarity_matrix.1D -polort -1 functional.nii'
    >>> res = corr.run()  # doctest: +SKIP
    """
    input_spec = AutoTcorrelateInputSpec
    output_spec = AFNICommandOutputSpec
    _cmd = '3dAutoTcorrelate'

    def _overload_extension(self, value, name=None):
        path, base, ext = split_filename(value)
        if ext.lower() not in ['.1d', '.1D', '.nii.gz', '.nii']:
            ext = ext + '.1D'
        return os.path.join(path, base + ext)