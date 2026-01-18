import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsInflate(FSCommand):
    """
    This program will inflate a cortical surface.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import MRIsInflate
    >>> inflate = MRIsInflate()
    >>> inflate.inputs.in_file = 'lh.pial'
    >>> inflate.inputs.no_save_sulc = True
    >>> inflate.cmdline # doctest: +SKIP
    'mris_inflate -no-save-sulc lh.pial lh.inflated'
    """
    _cmd = 'mris_inflate'
    input_spec = MRIsInflateInputSpec
    output_spec = MRIsInflateOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        if not self.inputs.no_save_sulc:
            outputs['out_sulc'] = os.path.abspath(self.inputs.out_sulc)
        return outputs