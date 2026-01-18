import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class DenoiseImageInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(2, 3, 4, argstr='-d %d', desc='This option forces the image to be treated as a specified-dimensional image. If not specified, the program tries to infer the dimensionality from the input image.')
    input_image = File(exists=True, argstr='-i %s', mandatory=True, desc='A scalar image is expected as input for noise correction.')
    noise_model = traits.Enum('Gaussian', 'Rician', argstr='-n %s', usedefault=True, desc='Employ a Rician or Gaussian noise model.')
    shrink_factor = traits.Int(default_value=1, usedefault=True, argstr='-s %s', desc='Running noise correction on large images can be time consuming. To lessen computation time, the input image can be resampled. The shrink factor, specified as a single integer, describes this resampling. Shrink factor = 1 is the default.')
    output_image = File(argstr='-o %s', name_source=['input_image'], hash_files=False, keep_extension=True, name_template='%s_noise_corrected', desc='The output consists of the noise corrected version of the input image.')
    save_noise = traits.Bool(False, mandatory=True, usedefault=True, desc='True if the estimated noise should be saved to file.', xor=['noise_image'])
    noise_image = File(name_source=['input_image'], hash_files=False, keep_extension=True, name_template='%s_noise', desc='Filename for the estimated noise.')
    verbose = traits.Bool(False, argstr='-v', desc='Verbose output.')