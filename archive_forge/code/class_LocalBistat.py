import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class LocalBistat(AFNICommand):
    """3dLocalBistat - computes statistics between 2 datasets, at each voxel,
    based on a local neighborhood of that voxel.

    For complete details, see the `3dLocalBistat Documentation.
    <https://afni.nimh.nih.gov/pub../pub/dist/doc/program_help/3dLocalBistat.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> bistat = afni.LocalBistat()
    >>> bistat.inputs.in_file1 = 'functional.nii'
    >>> bistat.inputs.in_file2 = 'structural.nii'
    >>> bistat.inputs.neighborhood = ('SPHERE', 1.2)
    >>> bistat.inputs.stat = 'pearson'
    >>> bistat.inputs.outputtype = 'NIFTI'
    >>> bistat.cmdline
    "3dLocalBistat -prefix functional_bistat.nii -nbhd 'SPHERE(1.2)' -stat pearson functional.nii structural.nii"
    >>> res = automask.run()  # doctest: +SKIP

    """
    _cmd = '3dLocalBistat'
    input_spec = LocalBistatInputSpec
    output_spec = AFNICommandOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'neighborhood' and value[0] == 'RECT':
            value = ('RECT', '%s,%s,%s' % value[1])
        return super(LocalBistat, self)._format_arg(name, spec, value)