import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRTransformInputSpec(MRTrix3BaseInputSpec):
    in_files = InputMultiPath(File(exists=True), argstr='%s', mandatory=True, position=-2, desc='Input images to be transformed')
    out_file = File(genfile=True, argstr='%s', position=-1, desc='Output image')
    invert = traits.Bool(argstr='-inverse', position=1, desc='Invert the specified transform before using it')
    linear_transform = File(exists=True, argstr='-linear %s', position=1, desc='Specify a linear transform to apply, in the form of a 3x4 or 4x4 ascii file. Note the standard reverse convention is used, where the transform maps points in the template image to the moving image. Note that the reverse convention is still assumed even if no -template image is supplied.')
    replace_transform = traits.Bool(argstr='-replace', position=1, desc='replace the current transform by that specified, rather than applying it to the current transform')
    transformation_file = File(exists=True, argstr='-transform %s', position=1, desc='The transform to apply, in the form of a 4x4 ascii file.')
    template_image = File(exists=True, argstr='-template %s', position=1, desc='Reslice the input image to match the specified template image.')
    reference_image = File(exists=True, argstr='-reference %s', position=1, desc='in case the transform supplied maps from the input image onto a reference image, use this option to specify the reference. Note that this implicitly sets the -replace option.')
    flip_x = traits.Bool(argstr='-flipx', position=1, desc="assume the transform is supplied assuming a coordinate system with the x-axis reversed relative to the MRtrix convention (i.e. x increases from right to left). This is required to handle transform matrices produced by FSL's FLIRT command. This is only used in conjunction with the -reference option.")
    quiet = traits.Bool(argstr='-quiet', position=1, desc='Do not display information messages or progress status.')
    debug = traits.Bool(argstr='-debug', position=1, desc='Display debugging messages.')