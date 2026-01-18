import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FindTheBiggest(FSLCommand):
    """
    Use FSL find_the_biggest for performing hard segmentation on
    the outputs of connectivity-based thresholding in probtrack.
    For complete details, see the `FDT
    Documentation. <http://www.fmrib.ox.ac.uk/fsl/fdt/fdt_biggest.html>`_

    Example
    -------

    >>> from nipype.interfaces import fsl
    >>> ldir = ['seeds_to_M1.nii', 'seeds_to_M2.nii']
    >>> fBig = fsl.FindTheBiggest(in_files=ldir, out_file='biggestSegmentation')
    >>> fBig.cmdline
    'find_the_biggest seeds_to_M1.nii seeds_to_M2.nii biggestSegmentation'

    """
    _cmd = 'find_the_biggest'
    input_spec = FindTheBiggestInputSpec
    output_spec = FindTheBiggestOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.out_file):
            self.inputs.out_file = self._gen_fname('biggestSegmentation', suffix='')
        return super(FindTheBiggest, self)._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = self._gen_fname('biggestSegmentation', suffix='')
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        else:
            return None