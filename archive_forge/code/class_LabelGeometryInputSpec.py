import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class LabelGeometryInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', usedefault=True, position=0, desc='image dimension (2 or 3)')
    label_image = File(argstr='%s', position=1, mandatory=True, desc='label image to use for extracting geometry measures')
    intensity_image = File(value='[]', exists=True, argstr='%s', mandatory=True, usedefault=True, position=2, desc='Intensity image to extract values from. This is an optional input')
    output_file = traits.Str(name_source=['label_image'], name_template='%s.csv', argstr='%s', position=3, desc='name of output file')