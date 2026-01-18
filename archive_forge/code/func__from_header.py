from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import numpy.typing as npt
def _from_header(header: str) -> List[Variable]:
    header = header.strip() + ',__dummy'
    entries = header.split(',')
    params = []
    start_idx = 0
    name = _get_base_name(entries[0])
    for i in range(0, len(entries) - 1):
        entry = entries[i]
        next_name = _get_base_name(entries[i + 1])
        if next_name != name:
            if ':' not in entry:
                dims = entry.split('.')[1:]
                if '.real' in entry or '.imag' in entry:
                    type = VariableType.COMPLEX
                    dims = dims[:-1]
                else:
                    type = VariableType.SCALAR
                params.append(Variable(name=name, start_idx=start_idx, end_idx=i + 1, dimensions=tuple(map(int, dims)), type=type, contents=[]))
            else:
                dims = entry.split(':')[0].split('.')[1:]
                munged_header = ','.join(dict.fromkeys(map(_munge_first_tuple, entries[start_idx:i + 1])))
                params.append(Variable(name=name, start_idx=start_idx, end_idx=i + 1, dimensions=tuple(map(int, dims)), type=VariableType.TUPLE, contents=_from_header(munged_header)))
            start_idx = i + 1
            name = next_name
    return params