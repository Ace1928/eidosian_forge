import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AverageAffineTransform(ANTSCommand):
    """
    Examples
    --------
    >>> from nipype.interfaces.ants import AverageAffineTransform
    >>> avg = AverageAffineTransform()
    >>> avg.inputs.dimension = 3
    >>> avg.inputs.transforms = ['trans.mat', 'func_to_struct.mat']
    >>> avg.inputs.output_affine_transform = 'MYtemplatewarp.mat'
    >>> avg.cmdline
    'AverageAffineTransform 3 MYtemplatewarp.mat trans.mat func_to_struct.mat'

    """
    _cmd = 'AverageAffineTransform'
    input_spec = AverageAffineTransformInputSpec
    output_spec = AverageAffineTransformOutputSpec

    def _format_arg(self, opt, spec, val):
        return super(AverageAffineTransform, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['affine_transform'] = os.path.abspath(self.inputs.output_affine_transform)
        return outputs