import os
import warnings
import pytest
from ....utils.filemanip import split_filename
from ... import base as nib
from ...base import traits, Undefined
from ....interfaces import fsl
from ...utility.wrappers import Function
from ....pipeline import Node
from ..specs import get_filecopy_info
class constrained_spec(nib.CommandLineInputSpec):
    in_file = nib.File(argstr='%s', position=1)
    threshold = traits.Float(argstr='%g', xor=['mask_file'], position=2)
    mask_file = nib.File(argstr='%s', name_source=['in_file'], name_template='%s_mask', keep_extension=True, xor=['threshold'], position=2)
    out_file1 = nib.File(argstr='%s', name_source=['in_file'], name_template='%s_out1', keep_extension=True, position=3)
    out_file2 = nib.File(argstr='%s', name_source=['in_file'], name_template='%s_out2', keep_extension=True, requires=['threshold'], position=4)