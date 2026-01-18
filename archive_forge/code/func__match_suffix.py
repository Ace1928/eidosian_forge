from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def _match_suffix(self, suffix: str, c_sig: str):
    args = c_sig.split(',')
    s2i = {'c': 1, 'd': 16}
    num_specs = 0
    sizes = []
    for i in range(len(args)):
        pos = suffix.find(str(i))
        if pos == -1:
            raise LinkerError(f'{suffix} is not a valid kernel suffix')
        pos += len(str(i))
        if self.arg_suffix.match(suffix, pos):
            num_specs += 1
            sizes.extend([None] * (i - len(sizes)))
            sizes.append(s2i[suffix[pos]])
            pos += 1
        if i < len(args) - 1:
            suffix = suffix[pos:]
        else:
            sizes.extend([None] * (len(args) - len(sizes)))
    return (num_specs, sizes)