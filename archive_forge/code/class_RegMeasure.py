import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegMeasure(NiftyRegCommand):
    """Interface for executable reg_measure from NiftyReg platform.

    Given two input images, compute the specified measure(s) of similarity

    `Source code <https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyreg
    >>> node = niftyreg.RegMeasure()
    >>> node.inputs.ref_file = 'im1.nii'
    >>> node.inputs.flo_file = 'im2.nii'
    >>> node.inputs.measure_type = 'lncc'
    >>> node.inputs.omp_core_val = 4
    >>> node.cmdline
    'reg_measure -flo im2.nii -lncc -omp 4 -out im2_lncc.txt -ref im1.nii'

    """
    _cmd = get_custom_path('reg_measure')
    input_spec = RegMeasureInputSpec
    output_spec = RegMeasureOutputSpec

    def _overload_extension(self, value, name=None):
        path, base, _ = split_filename(value)
        suffix = self.inputs.measure_type
        return os.path.join(path, '{0}_{1}.txt'.format(base, suffix))