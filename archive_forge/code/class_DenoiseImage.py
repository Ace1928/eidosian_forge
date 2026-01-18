import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class DenoiseImage(ANTSCommand):
    """
    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces.ants import DenoiseImage
    >>> denoise = DenoiseImage()
    >>> denoise.inputs.dimension = 3
    >>> denoise.inputs.input_image = 'im1.nii'
    >>> denoise.cmdline
    'DenoiseImage -d 3 -i im1.nii -n Gaussian -o im1_noise_corrected.nii -s 1'

    >>> denoise_2 = copy.deepcopy(denoise)
    >>> denoise_2.inputs.output_image = 'output_corrected_image.nii.gz'
    >>> denoise_2.inputs.noise_model = 'Rician'
    >>> denoise_2.inputs.shrink_factor = 2
    >>> denoise_2.cmdline
    'DenoiseImage -d 3 -i im1.nii -n Rician -o output_corrected_image.nii.gz -s 2'

    >>> denoise_3 = DenoiseImage()
    >>> denoise_3.inputs.input_image = 'im1.nii'
    >>> denoise_3.inputs.save_noise = True
    >>> denoise_3.cmdline
    'DenoiseImage -i im1.nii -n Gaussian -o [ im1_noise_corrected.nii, im1_noise.nii ] -s 1'

    """
    input_spec = DenoiseImageInputSpec
    output_spec = DenoiseImageOutputSpec
    _cmd = 'DenoiseImage'

    def _format_arg(self, name, trait_spec, value):
        if name == 'output_image' and (self.inputs.save_noise or isdefined(self.inputs.noise_image)):
            newval = '[ %s, %s ]' % (self._filename_from_source('output_image'), self._filename_from_source('noise_image'))
            return trait_spec.argstr % newval
        return super(DenoiseImage, self)._format_arg(name, trait_spec, value)