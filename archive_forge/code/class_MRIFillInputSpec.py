import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIFillInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', mandatory=True, exists=True, position=-2, desc='Input white matter file')
    out_file = File(argstr='%s', mandatory=True, exists=False, position=-1, desc='Output filled volume file name for MRIFill')
    segmentation = File(argstr='-segmentation %s', exists=True, desc='Input segmentation file for MRIFill')
    transform = File(argstr='-xform %s', exists=True, desc='Input transform file for MRIFill')
    log_file = File(argstr='-a %s', desc='Output log file for MRIFill')