import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Aparc2AsegOutputSpec(TraitedSpec):
    out_file = File(argstr='%s', desc='Output aseg file')