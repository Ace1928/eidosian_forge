import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class RenameInputSpec(DynamicTraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='file to rename')
    keep_ext = traits.Bool(desc='Keep in_file extension, replace non-extension component of name')
    format_string = Str(mandatory=True, desc='Python formatting string for output template')
    parse_string = Str(desc='Python regexp parse string to define replacement inputs')
    use_fullpath = traits.Bool(False, usedefault=True, desc='Use full path as input to regex parser')