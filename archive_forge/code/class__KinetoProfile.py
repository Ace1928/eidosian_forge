import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
class _KinetoProfile:
    """Low-level profiler wrap the autograd profile

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation (see ``export_memory_timeline``
            for more details).
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPS of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.

        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.

    .. note::
        This API is experimental and subject to change in the future.

        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.
    """

    def __init__(self, *, activities: Optional[Iterable[ProfilerActivity]]=None, record_shapes: bool=False, profile_memory: bool=False, with_stack: bool=False, with_flops: bool=False, with_modules: bool=False, experimental_config: Optional[_ExperimentalConfig]=None):
        self.activities = set(activities) if activities else supported_activities()
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.experimental_config = experimental_config
        self.profiler: Optional[prof.profile] = None
        self.mem_tl: Optional[MemoryProfileTimeline] = None
        self.use_device = None
        privateuse1_backend = _get_privateuse1_backend_name()
        if privateuse1_backend != 'privateuseone':
            self.use_device = privateuse1_backend

    def start(self):
        self.prepare_trace()
        self.start_trace()

    def stop(self):
        self.stop_trace()

    def prepare_trace(self):
        self.profiler = prof.profile(use_cuda=ProfilerActivity.CUDA in self.activities, use_cpu=ProfilerActivity.CPU in self.activities, use_mtia=ProfilerActivity.MTIA in self.activities, use_device=None, record_shapes=self.record_shapes, with_flops=self.with_flops, profile_memory=self.profile_memory, with_stack=self.with_stack, with_modules=self.with_modules, use_kineto=True, experimental_config=self.experimental_config)
        self.profiler._prepare_trace()

    def start_trace(self):
        assert self.profiler is not None
        self.profiler._start_trace()
        if self.profile_memory:
            self.add_metadata_json('profile_memory', '1')
        if self.with_stack:
            self.add_metadata_json('with_stack', '1')
        if self.record_shapes:
            self.add_metadata_json('record_shapes', '1')
        if self.with_modules:
            self.add_metadata_json('with_modules', '1')
        if self.with_flops:
            self.add_metadata_json('with_flops', '1')
        if kineto_available():
            dist_info = self._get_distributed_info()
            if dist_info:
                self.add_metadata_json('distributedInfo', json.dumps(dist_info))
            if hasattr(torch, '_inductor'):
                import torch._inductor.config as inductor_config
                if inductor_config.triton.cudagraphs:
                    os.environ['DISABLE_CUPTI_LAZY_REINIT'] = '1'
                    self.add_metadata_json('DISABLE_CUPTI_LAZY_REINIT', '1')
                    os.environ['TEARDOWN_CUPTI'] = '0'

    def stop_trace(self):
        assert self.profiler is not None
        self.profiler.__exit__(None, None, None)

    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format.
        """
        assert self.profiler
        if path.endswith('.gz'):
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
            fp.close()
            retvalue = self.profiler.export_chrome_trace(fp.name)
            with open(fp.name) as fin:
                with gzip.open(path, 'wt') as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
            return retvalue
        else:
            return self.profiler.export_chrome_trace(path)

    def export_stacks(self, path: str, metric: str='self_cpu_time_total'):
        """Save stack traces in a file in a format suitable for visualization.

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_cuda_time_total"

        .. note::
            Example of using FlameGraph tool:

            - git clone https://github.com/brendangregg/FlameGraph
            - cd FlameGraph
            - ./flamegraph.pl --title "CPU time" --countname "us." profiler.stacks > perf_viz.svg
        """
        assert self.profiler
        return self.profiler.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape: bool=False, group_by_stack_n: int=0):
        """Averages events, grouping them by operator name and (optionally) input shapes and
        stack.

        .. note::
            To use shape/stack functionality make sure to set record_shapes/with_stack
            when creating profiler context manager.
        """
        assert self.profiler
        return self.profiler.key_averages(group_by_input_shape, group_by_stack_n)

    def events(self):
        """
        Returns the list of unaggregated profiler events,
        to be used in the trace callback or after the profiling is finished
        """
        assert self.profiler
        return self.profiler.function_events

    def add_metadata(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
        wrapped_value = '"' + value.replace('"', '\\"') + '"'
        torch.autograd._add_metadata_json(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
        torch.autograd._add_metadata_json(key, value)

    def _get_distributed_info(self):
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return None
        return {'backend': dist.get_backend(), 'rank': dist.get_rank(), 'world_size': dist.get_world_size()}

    def _memory_profile(self) -> MemoryProfile:
        required = ('record_shapes', 'profile_memory', 'with_stack')
        missing = [f'{i}=True' for i in required if not getattr(self, i)]
        if missing:
            raise ValueError(f'{', '.join(missing)} required for memory profiling.')
        assert self.profiler is not None and self.profiler.kineto_results is not None
        return MemoryProfile(self.profiler.kineto_results)

    def export_memory_timeline(self, path: str, device: Optional[str]=None) -> None:
        """Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.
        """
        if device is None and self.use_device and (self.use_device != 'cuda'):
            device = self.use_device + ':0'
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mem_tl = MemoryProfileTimeline(self._memory_profile())
        if path.endswith('.html'):
            self.mem_tl.export_memory_timeline_html(path, device)
        elif path.endswith('.gz'):
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
            fp.close()
            if path.endswith('raw.json.gz'):
                self.mem_tl.export_memory_timeline_raw(fp.name, device)
            else:
                self.mem_tl.export_memory_timeline(fp.name, device)
            with open(fp.name) as fin:
                with gzip.open(path, 'wt') as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
        else:
            self.mem_tl.export_memory_timeline(path, device)