import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AverageImages(ANTSCommand):
    """
    Examples
    --------
    >>> from nipype.interfaces.ants import AverageImages
    >>> avg = AverageImages()
    >>> avg.inputs.dimension = 3
    >>> avg.inputs.output_average_image = "average.nii.gz"
    >>> avg.inputs.normalize = True
    >>> avg.inputs.images = ['rc1s1.nii', 'rc1s1.nii']
    >>> avg.cmdline
    'AverageImages 3 average.nii.gz 1 rc1s1.nii rc1s1.nii'
    """
    _cmd = 'AverageImages'
    input_spec = AverageImagesInputSpec
    output_spec = AverageImagesOutputSpec

    def _format_arg(self, opt, spec, val):
        return super(AverageImages, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_average_image'] = os.path.realpath(self.inputs.output_average_image)
        return outputs