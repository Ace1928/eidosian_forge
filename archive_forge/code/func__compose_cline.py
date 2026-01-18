from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _compose_cline(self, i: int, icol: int) -> str:
    """
        Create clines after multirow-blocks are finished.
        """
    lst = []
    for cl in self.clinebuf:
        if cl[0] == i:
            lst.append(f'\n\\cline{{{cl[1]:d}-{icol:d}}}')
            self.clinebuf = [x for x in self.clinebuf if x[0] != i]
    return ''.join(lst)