from __future__ import annotations
import builtins
import time
from typing import Dict
from ..testing import do_bench
from .jit import KernelInterface
def _post_hook(args):
    for i, j in enumerate(self.restore_idx):
        args[j].copy_(self.restore_copies[i])
    self.restore_copies = []