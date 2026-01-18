import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Apas2AsegInputSpec(FSTraitedSpec):
    in_file = File(argstr='--i %s', mandatory=True, exists=True, desc='Input aparc+aseg.mgz')
    out_file = File(argstr='--o %s', mandatory=True, desc='Output aseg file')