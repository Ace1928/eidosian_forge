import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsInflateInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', position=-2, mandatory=True, exists=True, copyfile=True, desc='Input file for MRIsInflate')
    out_file = File(argstr='%s', position=-1, exists=False, name_source=['in_file'], name_template='%s.inflated', hash_files=False, keep_extension=True, desc='Output file for MRIsInflate')
    out_sulc = File(exists=False, xor=['no_save_sulc'], desc='Output sulc file')
    no_save_sulc = traits.Bool(argstr='-no-save-sulc', xor=['out_sulc'], desc='Do not save sulc file as output')