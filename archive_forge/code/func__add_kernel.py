from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def _add_kernel(self, name: str, ker: KernelLinkerMeta):
    if name in self.kernels:
        last: KernelLinkerMeta = self.kernels[name][-1]
        for cur, new_ in zip(last.arg_ctypes, ker.arg_ctypes):
            if cur != new_:
                raise LinkerError(f'Mismatched signature for kernel {name}: \n\texisting sig is: {','.join(last.arg_ctypes)}\n\tcurrent is: {','.join(ker.arg_ctypes)}')
    self.kernels[name].append(ker)