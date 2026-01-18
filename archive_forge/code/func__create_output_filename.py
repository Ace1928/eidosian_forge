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
def _create_output_filename(self, filename: str) -> Path:
    """
        Returns where to write a file with desired filename.
        Handles the case where we are in distributed settings, or when
        we need to output the same file multiple times (eg if a profiler
        runs for several steps)
        """
    if self.worker_name != '':
        file = Path(filename)
        folder = self.output_dir / file.stem
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.done_steps:06}_{self.worker_name}{file.suffix}'
    return self.output_dir / f'{self.done_steps:06}_{filename}'