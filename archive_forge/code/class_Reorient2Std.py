import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class Reorient2Std(FSLCommand):
    """fslreorient2std is a tool for reorienting the image to match the
    approximate orientation of the standard template images (MNI152).


    Examples
    --------

    >>> reorient = Reorient2Std()
    >>> reorient.inputs.in_file = "functional.nii"
    >>> res = reorient.run() # doctest: +SKIP


    """
    _cmd = 'fslreorient2std'
    input_spec = Reorient2StdInputSpec
    output_spec = Reorient2StdOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_fname(self.inputs.in_file, suffix='_reoriented')
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_filename('out_file')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs