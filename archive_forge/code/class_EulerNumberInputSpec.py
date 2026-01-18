import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class EulerNumberInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', position=-1, mandatory=True, exists=True, desc='Input file for EulerNumber')