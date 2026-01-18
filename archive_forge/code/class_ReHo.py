import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ReHo(AFNICommandBase):
    """Compute regional homogeneity for a given neighbourhood.l,
    based on a local neighborhood of that voxel.

    For complete details, see the `3dReHo Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dReHo.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> reho = afni.ReHo()
    >>> reho.inputs.in_file = 'functional.nii'
    >>> reho.inputs.out_file = 'reho.nii.gz'
    >>> reho.inputs.neighborhood = 'vertices'
    >>> reho.cmdline
    '3dReHo -prefix reho.nii.gz -inset functional.nii -nneigh 27'
    >>> res = reho.run()  # doctest: +SKIP

    """
    _cmd = '3dReHo'
    input_spec = ReHoInputSpec
    output_spec = ReHoOutputSpec

    def _list_outputs(self):
        outputs = super(ReHo, self)._list_outputs()
        if self.inputs.label_set:
            outputs['out_vals'] = outputs['out_file'] + '_ROI_reho.vals'
        return outputs

    def _format_arg(self, name, spec, value):
        _neigh_dict = {'faces': 7, 'edges': 19, 'vertices': 27}
        if name == 'neighborhood':
            value = _neigh_dict[value]
        return super(ReHo, self)._format_arg(name, spec, value)