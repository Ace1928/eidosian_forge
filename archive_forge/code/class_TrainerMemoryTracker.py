import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless `psutil` is available. Install with `pip install psutil`.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example :

    ```python
    self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
    self._memory_tracker.start()
    # code ...
    metrics = {"train_runtime": 10.5}
    self._memory_tracker.stop_and_update_metrics(metrics)
    ```

    At the moment GPU tracking is only for `pytorch`, but can be extended to support `tensorflow`.

    To understand this class' intricacies please read the documentation of [`~Trainer.log_metrics`].
    """
    stages = {'__init__': 'init', 'train': 'train', '_inner_training_loop': 'train', 'evaluate': 'eval', 'predict': 'test'}

    def __init__(self, skip_memory_metrics=False):
        self.skip_memory_metrics = skip_memory_metrics
        if not is_psutil_available():
            self.skip_memory_metrics = True
        if self.skip_memory_metrics:
            return
        import psutil
        if is_torch_cuda_available():
            import torch
            self.torch = torch
            self.gpu = {}
        elif is_torch_mps_available():
            import torch
            self.torch = torch
            self.gpu = {}
        elif is_torch_xpu_available():
            import torch
            self.torch = torch
            self.gpu = {}
        elif is_torch_npu_available():
            import torch
            self.torch = torch
            self.gpu = {}
        else:
            self.torch = None
        self.process = psutil.Process()
        self.cur_stage = None
        self.cpu = {}
        self.init_reported = False

    def derive_stage(self):
        """derives the stage/caller name automatically"""
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        if caller in self.stages:
            return self.stages[caller]
        else:
            raise ValueError(f'was called from {caller}, but only expect to be called from one of {self.stages.keys()}')

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1
        while True:
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)
            if not self.peak_monitoring:
                break

    def start(self):
        """start tracking for the caller's stage"""
        if self.skip_memory_metrics:
            return
        stage = self.derive_stage()
        if self.cur_stage is not None and self.cur_stage != stage:
            return
        self.cur_stage = stage
        gc.collect()
        if self.torch is not None:
            if torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()
                self.torch.cuda.empty_cache()
            elif is_torch_xpu_available():
                self.torch.xpu.reset_peak_memory_stats()
                self.torch.xpu.empty_cache()
            elif is_torch_npu_available():
                self.torch.npu.reset_peak_memory_stats()
                self.torch.npu.empty_cache()
        if self.torch is not None:
            if torch.cuda.is_available():
                self.gpu_mem_used_at_start = self.torch.cuda.memory_allocated()
            elif is_torch_xpu_available():
                self.gpu_mem_used_at_start = self.torch.xpu.memory_allocated()
            elif is_torch_npu_available():
                self.gpu_mem_used_at_start = self.torch.npu.memory_allocated()
        self.cpu_mem_used_at_start = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def stop(self, stage):
        """stop tracking for the passed stage"""
        if self.cur_stage is not None and self.cur_stage != stage:
            return
        self.peak_monitoring = False
        gc.collect()
        if self.torch is not None:
            if torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            elif is_torch_xpu_available():
                self.torch.xpu.empty_cache()
            elif is_torch_npu_available():
                self.torch.npu.empty_cache()
        if self.torch is not None:
            if torch.cuda.is_available():
                self.gpu_mem_used_now = self.torch.cuda.memory_allocated()
                self.gpu_mem_used_peak = self.torch.cuda.max_memory_allocated()
            elif is_torch_xpu_available():
                self.gpu_mem_used_now = self.torch.xpu.memory_allocated()
                self.gpu_mem_used_peak = self.torch.xpu.max_memory_allocated()
            elif is_torch_npu_available():
                self.gpu_mem_used_now = self.torch.npu.memory_allocated()
                self.gpu_mem_used_peak = self.torch.npu.max_memory_allocated()
            else:
                raise ValueError('No available GPU device found!')
            self.gpu[self.cur_stage] = {'begin': self.gpu_mem_used_at_start, 'end': self.gpu_mem_used_now, 'alloc': self.gpu_mem_used_now - self.gpu_mem_used_at_start, 'peaked': max(0, self.gpu_mem_used_peak - self.gpu_mem_used_now)}
        self.cpu_mem_used_now = self.cpu_mem_used()
        self.cpu[self.cur_stage] = {'begin': self.cpu_mem_used_at_start, 'end': self.cpu_mem_used_now, 'alloc': self.cpu_mem_used_now - self.cpu_mem_used_at_start, 'peaked': max(0, self.cpu_mem_used_peak - self.cpu_mem_used_now)}
        self.cur_stage = None

    def update_metrics(self, stage, metrics):
        """updates the metrics"""
        if self.skip_memory_metrics:
            return
        if self.cur_stage is not None and self.cur_stage != stage:
            return
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, 'init')
            self.init_reported = True
        for stage in stages:
            for t in ['alloc', 'peaked']:
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f'{stage}_mem_cpu_{t}_delta'] = self.cpu[stage][t]
                if self.torch is not None and stage in self.gpu and (t in self.gpu[stage]):
                    metrics[f'{stage}_mem_gpu_{t}_delta'] = self.gpu[stage][t]
        if stages[0] == 'init':
            metrics['before_init_mem_cpu'] = self.cpu['init']['begin']
            if self.torch is not None:
                metrics['before_init_mem_gpu'] = self.gpu['init']['begin']

    def stop_and_update_metrics(self, metrics=None):
        """combine stop and metrics update in one call for simpler code"""
        if self.skip_memory_metrics:
            return
        stage = self.derive_stage()
        self.stop(stage)
        if metrics is not None:
            self.update_metrics(stage, metrics)