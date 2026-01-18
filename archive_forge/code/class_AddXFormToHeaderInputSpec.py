import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class AddXFormToHeaderInputSpec(FSTraitedSpec):
    in_file = File(exists=True, mandatory=True, position=-2, argstr='%s', desc='input volume')
    transform = File(exists=False, mandatory=True, position=-3, argstr='%s', desc='xfm file')
    out_file = File('output.mgz', position=-1, argstr='%s', usedefault=True, desc='output volume')
    copy_name = traits.Bool(argstr='-c', desc='do not try to load the xfmfile, just copy name')
    verbose = traits.Bool(argstr='-v', desc='be verbose')