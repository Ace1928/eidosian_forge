import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AffineInitializer(ANTSCommand):
    """
    Initialize an affine transform (as in antsBrainExtraction.sh)

    >>> from nipype.interfaces.ants import AffineInitializer
    >>> init = AffineInitializer()
    >>> init.inputs.fixed_image = 'fixed1.nii'
    >>> init.inputs.moving_image = 'moving1.nii'
    >>> init.cmdline
    'antsAffineInitializer 3 fixed1.nii moving1.nii transform.mat 15.000000 0.100000 0 10'

    """
    _cmd = 'antsAffineInitializer'
    input_spec = AffineInitializerInputSpec
    output_spec = AffineInitializerOutputSpec

    def _list_outputs(self):
        return {'out_file': os.path.abspath(self.inputs.out_file)}