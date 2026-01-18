import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ExtractMainComponentInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=1, desc='input surface file')
    out_file = File(name_template='%s.maincmp', name_source='in_file', argstr='%s', position=2, desc='surface containing main component')