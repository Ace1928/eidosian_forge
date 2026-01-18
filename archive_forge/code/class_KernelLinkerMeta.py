from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
@dataclass
class KernelLinkerMeta:
    orig_kernel_name: str
    arg_names: Sequence[str]
    arg_ctypes: Sequence[str]
    sizes: Sequence[Union[int, None]]
    sig_hash: str
    triton_suffix: str
    suffix: str
    num_specs: int
    ' number of specialized arguments '