import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class EstimateResponseForSH(CommandLine):
    """
    Estimates the fibre response function for use in spherical deconvolution.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> estresp = mrt.EstimateResponseForSH()
    >>> estresp.inputs.in_file = 'dwi.mif'
    >>> estresp.inputs.mask_image = 'dwi_WMProb.mif'
    >>> estresp.inputs.encoding_file = 'encoding.txt'
    >>> estresp.run()                                   # doctest: +SKIP
    """
    _cmd = 'estimate_response'
    input_spec = EstimateResponseForSHInputSpec
    output_spec = EstimateResponseForSHOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['response'] = self.inputs.out_filename
        if not isdefined(outputs['response']):
            outputs['response'] = op.abspath(self._gen_outfilename())
        else:
            outputs['response'] = op.abspath(outputs['response'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_ER.txt'