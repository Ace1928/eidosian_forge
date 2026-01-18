from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
def _assert_nested_dict_equal(a: dict, b: dict):
    if len(a) != len(b):
        return False
    for key in a:
        if key in b:
            if isinstance(a[key], dict):
                if not _assert_nested_dict_equal(a[key], b[key]):
                    return False
            elif isinstance(a[key], np.ndarray):
                if not np.all(a[key] == b[key]):
                    return False
            elif a[key] != b[key]:
                return False
        else:
            return False
    return True