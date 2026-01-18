import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class SelectInputSpec(BaseInterfaceInputSpec):
    inlist = InputMultiPath(traits.Any, mandatory=True, desc='list of values to choose from')
    index = InputMultiPath(traits.Int, mandatory=True, desc='0-based indices of values to choose')