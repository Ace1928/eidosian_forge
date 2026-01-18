import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MakeDyadicVectors(FSLCommand):
    """Create vector volume representing mean principal diffusion direction
    and its uncertainty (dispersion)"""
    _cmd = 'make_dyadic_vectors'
    input_spec = MakeDyadicVectorsInputSpec
    output_spec = MakeDyadicVectorsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['dyads'] = self._gen_fname(self.inputs.output)
        outputs['dispersion'] = self._gen_fname(self.inputs.output, suffix='_dispersion')
        return outputs