import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class FSL2MRTrix(BaseInterface):
    """
    Converts separate b-values and b-vectors from text files (FSL style) into a
    4xN text file in which each line is in the format [ X Y Z b ], where [ X Y Z ]
    describe the direction of the applied gradient, and b gives the
    b-value in units (1000 s/mm^2).

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> fsl2mrtrix = mrt.FSL2MRTrix()
    >>> fsl2mrtrix.inputs.bvec_file = 'bvecs'
    >>> fsl2mrtrix.inputs.bval_file = 'bvals'
    >>> fsl2mrtrix.inputs.invert_y = True
    >>> fsl2mrtrix.run()                                # doctest: +SKIP
    """
    input_spec = FSL2MRTrixInputSpec
    output_spec = FSL2MRTrixOutputSpec

    def _run_interface(self, runtime):
        encoding = concat_files(self.inputs.bvec_file, self.inputs.bval_file, self.inputs.invert_x, self.inputs.invert_y, self.inputs.invert_z)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['encoding_file'] = op.abspath(self._gen_filename('out_encoding_file'))
        return outputs

    def _gen_filename(self, name):
        if name == 'out_encoding_file':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, bvec, _ = split_filename(self.inputs.bvec_file)
        _, bval, _ = split_filename(self.inputs.bval_file)
        return bvec + '_' + bval + '.txt'