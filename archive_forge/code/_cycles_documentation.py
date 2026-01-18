import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging

    Install a warning that reports whenever a cycle that is holding CUDA memory is observed.

    The warning produces an .html file that visualizes the cycle,
    and links it to the stack frame that allocted the CUDA tensor.

    Reference cycles are freed by the cycle collector rather than being cleaned up
    when the objects in the cycle first become unreachable. If a cycle points to a tensor,
    the CUDA memory for that tensor will not be freed until garbage collection runs.
    Accumulation of CUDA allocations can lead to out of memory errors (OOMs), as well as
    non-deterministic allocation behavior which is harder to debug.
    