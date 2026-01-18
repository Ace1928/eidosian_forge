import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PowerSpectrum(FSLCommand):
    """Use FSL PowerSpectrum command for power spectrum estimation.

    Examples
    --------

    >>> from nipype.interfaces import fsl
    >>> pspec = fsl.PowerSpectrum()
    >>> pspec.inputs.in_file = 'functional.nii'
    >>> res = pspec.run() # doctest: +SKIP


    """
    _cmd = 'fslpspec'
    input_spec = PowerSpectrumInputSpec
    output_spec = PowerSpectrumOutputSpec

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = self._gen_fname(self.inputs.in_file, suffix='_ps')
        return out_file

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None