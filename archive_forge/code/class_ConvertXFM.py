import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ConvertXFM(FSLCommand):
    """Use the FSL utility convert_xfm to modify FLIRT transformation matrices.

    Examples
    --------

    >>> import nipype.interfaces.fsl as fsl
    >>> invt = fsl.ConvertXFM()
    >>> invt.inputs.in_file = "flirt.mat"
    >>> invt.inputs.invert_xfm = True
    >>> invt.inputs.out_file = 'flirt_inv.mat'
    >>> invt.cmdline
    'convert_xfm -omat flirt_inv.mat -inverse flirt.mat'


    """
    _cmd = 'convert_xfm'
    input_spec = ConvertXFMInputSpec
    output_spec = ConvertXFMOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outfile = self.inputs.out_file
        if not isdefined(outfile):
            _, infile1, _ = split_filename(self.inputs.in_file)
            if self.inputs.invert_xfm:
                outfile = fname_presuffix(infile1, suffix='_inv.mat', newpath=os.getcwd(), use_ext=False)
            elif self.inputs.concat_xfm:
                _, infile2, _ = split_filename(self.inputs.in_file2)
                outfile = fname_presuffix('%s_%s' % (infile1, infile2), suffix='.mat', newpath=os.getcwd(), use_ext=False)
            else:
                outfile = fname_presuffix(infile1, suffix='_fix.mat', newpath=os.getcwd(), use_ext=False)
        outputs['out_file'] = os.path.abspath(outfile)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['out_file']
        return None