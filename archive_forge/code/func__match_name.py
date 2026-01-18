from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def _match_name(self, ker_name: str):
    m = self.kernel_name.match(ker_name)
    if _exists(m):
        name, sig_hash, suffix = (m.group(1), m.group(2), m.group(3))
        return (name, sig_hash, suffix)
    raise LinkerError(f'{ker_name} is not a valid kernel name')