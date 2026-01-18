import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Seg(AFNICommandBase):
    """3dSeg segments brain volumes into tissue classes. The program allows
    for adding a variety of global and voxelwise priors. However for the
    moment, only mixing fractions and MRF are documented.

    For complete details, see the `3dSeg Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dSeg.html>`_

    Examples
    --------
    >>> from nipype.interfaces.afni import preprocess
    >>> seg = preprocess.Seg()
    >>> seg.inputs.in_file = 'structural.nii'
    >>> seg.inputs.mask = 'AUTO'
    >>> seg.cmdline
    '3dSeg -mask AUTO -anat structural.nii'
    >>> res = seg.run()  # doctest: +SKIP

    """
    _cmd = '3dSeg'
    input_spec = SegInputSpec
    output_spec = AFNICommandOutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        import glob
        outputs = self._outputs()
        if isdefined(self.inputs.prefix):
            outfile = os.path.join(os.getcwd(), self.inputs.prefix, 'Classes+*.BRIK')
        else:
            outfile = os.path.join(os.getcwd(), 'Segsy', 'Classes+*.BRIK')
        outputs.out_file = glob.glob(outfile)[0]
        return outputs