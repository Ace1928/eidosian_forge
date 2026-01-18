import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SwapDimensions(FSLCommand):
    """Use fslswapdim to alter the orientation of an image.

    This interface accepts a three-tuple corresponding to the new
    orientation.  You may either provide dimension ids in the form of
    (-)x, (-)y, or (-z), or nifti-syle dimension codes
    (RL, LR, AP, PA, IS, SI).

    """
    _cmd = 'fslswapdim'
    input_spec = SwapDimensionsInputSpec
    output_spec = SwapDimensionsOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file, suffix='_newdims')
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['out_file']
        return None