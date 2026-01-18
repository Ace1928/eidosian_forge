import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class ROIStats(AFNICommandBase):
    """Display statistics over masked regions

    For complete details, see the `3dROIstats Documentation
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dROIstats.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> roistats = afni.ROIStats()
    >>> roistats.inputs.in_file = 'functional.nii'
    >>> roistats.inputs.mask_file = 'skeleton_mask.nii.gz'
    >>> roistats.inputs.stat = ['mean', 'median', 'voxels']
    >>> roistats.inputs.nomeanout = True
    >>> roistats.cmdline
    '3dROIstats -mask skeleton_mask.nii.gz -nomeanout -nzmean -nzmedian -nzvoxels functional.nii > functional_roistat.1D'
    >>> res = roistats.run()  # doctest: +SKIP

    """
    _cmd = '3dROIstats'
    _terminal_output = 'allatonce'
    input_spec = ROIStatsInputSpec
    output_spec = ROIStatsOutputSpec

    def _format_arg(self, name, trait_spec, value):
        _stat_dict = {'mean': '-nzmean', 'median': '-nzmedian', 'mode': '-nzmode', 'minmax': '-nzminmax', 'sigma': '-nzsigma', 'voxels': '-nzvoxels', 'sum': '-nzsum', 'summary': '-summary', 'zerominmax': '-minmax', 'zeromedian': '-median', 'zerosigma': '-sigma', 'zeromode': '-mode'}
        if name == 'stat':
            value = [_stat_dict[v] for v in value]
        return super(ROIStats, self)._format_arg(name, trait_spec, value)