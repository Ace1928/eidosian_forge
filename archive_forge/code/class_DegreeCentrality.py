import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class DegreeCentrality(AFNICommand):
    """Performs degree centrality on a dataset using a given maskfile
    via 3dDegreeCentrality

    For complete details, see the `3dDegreeCentrality Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDegreeCentrality.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> degree = afni.DegreeCentrality()
    >>> degree.inputs.in_file = 'functional.nii'
    >>> degree.inputs.mask = 'mask.nii'
    >>> degree.inputs.sparsity = 1 # keep the top one percent of connections
    >>> degree.inputs.out_file = 'out.nii'
    >>> degree.cmdline
    '3dDegreeCentrality -mask mask.nii -prefix out.nii -sparsity 1.000000 functional.nii'
    >>> res = degree.run()  # doctest: +SKIP

    """
    _cmd = '3dDegreeCentrality'
    input_spec = DegreeCentralityInputSpec
    output_spec = DegreeCentralityOutputSpec

    def _list_outputs(self):
        outputs = super(DegreeCentrality, self)._list_outputs()
        if self.inputs.oned_file:
            outputs['oned_file'] = os.path.abspath(self.inputs.oned_file)
        return outputs