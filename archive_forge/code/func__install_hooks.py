import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
def _install_hooks(self) -> None:
    self.hooks_refcount += 1
    if self.hooks:
        return
    module = self.module()
    if module is None:
        return
    for name, sub_mod in module.named_modules():
        if name == '':
            continue
        name = name.split('.')[-1]
        self.hooks += [sub_mod.register_forward_pre_hook(self._enter_module_hook(name)), sub_mod.register_forward_hook(self._exit_module_hook(name))]