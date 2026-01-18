import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def _read_datafile(spg, spacegroup, setting, f):
    if isinstance(spacegroup, int):
        pass
    elif isinstance(spacegroup, str):
        spacegroup = ' '.join(spacegroup.strip().split())
        compact_spacegroup = ''.join(spacegroup.split())
    else:
        raise SpacegroupValueError('`spacegroup` must be of type int or str')
    while True:
        line1, line2 = _skip_to_nonblank(f, spacegroup, setting)
        _no, _symbol = line1.strip().split(None, 1)
        _symbol = format_symbol(_symbol)
        compact_symbol = ''.join(_symbol.split())
        _setting = int(line2.strip().split()[1])
        _no = int(_no)
        if isinstance(spacegroup, int) and _no == spacegroup and (_setting == setting) or ((isinstance(spacegroup, str) and compact_symbol == compact_spacegroup) and (setting is None or _setting == setting)):
            _read_datafile_entry(spg, _no, _symbol, _setting, f)
            break
        else:
            _skip_to_blank(f, spacegroup, setting)