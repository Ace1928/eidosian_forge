import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class BrainExtractionInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='-d %d', usedefault=True, desc='image dimension (2 or 3)')
    anatomical_image = File(exists=True, argstr='-a %s', desc='Structural image, typically T1.  If more than one anatomical image is specified, subsequently specified images are used during the segmentation process.  However, only the first image is used in the registration of priors. Our suggestion would be to specify the T1 as the first image. Anatomical template created using e.g. LPBA40 data set with buildtemplateparallel.sh in ANTs.', mandatory=True)
    brain_template = File(exists=True, argstr='-e %s', desc='Anatomical template created using e.g. LPBA40 data set with buildtemplateparallel.sh in ANTs.', mandatory=True)
    brain_probability_mask = File(exists=True, argstr='-m %s', desc='Brain probability mask created using e.g. LPBA40 data set which have brain masks defined, and warped to anatomical template and averaged resulting in a probability image.', copyfile=False, mandatory=True)
    out_prefix = traits.Str('highres001_', argstr='-o %s', usedefault=True, desc='Prefix that is prepended to all output files')
    extraction_registration_mask = File(exists=True, argstr='-f %s', desc='Mask (defined in the template space) used during registration for brain extraction. To limit the metric computation to a specific region.')
    image_suffix = traits.Str('nii.gz', desc='any of standard ITK formats, nii.gz is default', argstr='-s %s', usedefault=True)
    use_random_seeding = traits.Enum(0, 1, argstr='-u %d', desc='Use random number generated from system clock in Atropos (default = 1)')
    keep_temporary_files = traits.Int(argstr='-k %d', desc='Keep brain extraction/segmentation warps, etc (default = 0).')
    use_floatingpoint_precision = traits.Enum(0, 1, argstr='-q %d', desc='Use floating point precision in registrations (default = 0)')
    debug = traits.Bool(argstr='-z 1', desc='If > 0, runs a faster version of the script. Only for testing. Implies -u 0. Requires single thread computation for complete reproducibility.')