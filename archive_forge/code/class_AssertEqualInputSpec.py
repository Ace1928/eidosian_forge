import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class AssertEqualInputSpec(BaseInterfaceInputSpec):
    volume1 = File(exists=True, mandatory=True)
    volume2 = File(exists=True, mandatory=True)