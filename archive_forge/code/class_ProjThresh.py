import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProjThresh(FSLCommand):
    """Use FSL proj_thresh for thresholding some outputs of probtrack
    For complete details, see the FDT Documentation
    <http://www.fmrib.ox.ac.uk/fsl/fdt/fdt_thresh.html>

    Example
    -------

    >>> from nipype.interfaces import fsl
    >>> ldir = ['seeds_to_M1.nii', 'seeds_to_M2.nii']
    >>> pThresh = fsl.ProjThresh(in_files=ldir, threshold=3)
    >>> pThresh.cmdline
    'proj_thresh seeds_to_M1.nii seeds_to_M2.nii 3'

    """
    _cmd = 'proj_thresh'
    input_spec = ProjThreshInputSpec
    output_spec = ProjThreshOuputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_files'] = []
        for name in self.inputs.in_files:
            cwd, base_name = os.path.split(name)
            outputs['out_files'].append(self._gen_fname(base_name, cwd=cwd, suffix='_proj_seg_thr_{}'.format(self.inputs.threshold)))
        return outputs