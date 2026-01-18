import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def _pattern(self, pData):
    data = pData
    if '' in data and len(data.keys()) == 1:
        return None
    alt = []
    cc = []
    q = 0
    for char in sorted(data.keys()):
        if isinstance(data[char], dict):
            try:
                recurse = self._pattern(data[char])
                alt.append(self.quote(char) + recurse)
            except Exception:
                cc.append(self.quote(char))
        else:
            q = 1
    cconly = not len(alt) > 0
    if len(cc) > 0:
        if len(cc) == 1:
            alt.append(cc[0])
        else:
            alt.append('[' + ''.join(cc) + ']')
    if len(alt) == 1:
        result = alt[0]
    else:
        result = '(?:' + '|'.join(alt) + ')'
    if q:
        if cconly:
            result += '?'
        else:
            result = f'(?:{result})?'
    return result