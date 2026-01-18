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
class MemSnapshotsProfiler:
    """Profiler that captures memory traces for allocation and deallocation of memory for
    tensors.
    """

    def __init__(self, main_profiler: '_Profiler') -> None:
        self.main_profiler = main_profiler
        self.enabled = False

    @property
    def _has_trace_plot(self) -> bool:
        return hasattr(torch.cuda._memory_viz, 'trace_plot')

    def __enter__(self):
        if not self._has_trace_plot:
            return
        self.enabled = True
        torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100000, trace_alloc_record_context=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._has_trace_plot:
            self.main_profiler.summary.append(('MemTrace', '(not available with your Pytorch version)'))
            return
        assert self.enabled
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(False)
        if all((len(t) == 0 for t in snapshot['device_traces'])):
            self.main_profiler.summary.append(('MemTrace', '(no allocation recorded)'))
            return
        filename = self.main_profiler._create_output_filename('memory_trace_plot.html')
        self.main_profiler.summary.append(('MemTrace', filename))
        with open(filename, 'w+') as fd:
            fd.write(torch.cuda._memory_viz.trace_plot(snapshot, device=None, plot_segments=False))

    def step(self) -> None:
        pass