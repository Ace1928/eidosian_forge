import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class MathsInput(CommandLineInputSpec):
    """Input Spec for seg_maths interfaces."""
    in_file = File(position=2, argstr='%s', exists=True, mandatory=True, desc='image to operate on')
    out_file = File(name_source=['in_file'], name_template='%s', position=-2, argstr='%s', desc='image to write')
    desc = 'datatype to use for output (default uses input type)'
    output_datatype = traits.Enum('float', 'char', 'int', 'short', 'double', 'input', position=-3, argstr='-odt %s', desc=desc)