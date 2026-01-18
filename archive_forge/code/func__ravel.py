import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
def _ravel(in_val):
    if not isinstance(in_val, list):
        return in_val
    flat_list = []
    for val in in_val:
        raveled_val = _ravel(val)
        if isinstance(raveled_val, list):
            flat_list.extend(raveled_val)
        else:
            flat_list.append(raveled_val)
    return flat_list