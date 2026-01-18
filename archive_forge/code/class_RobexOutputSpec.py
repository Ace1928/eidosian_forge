import os
from pathlib import Path
from nipype.interfaces.base import (
from nipype.utils.filemanip import split_filename
class RobexOutputSpec(TraitedSpec):
    out_file = File(desc='Output volume')
    out_mask = File(desc='Output mask')