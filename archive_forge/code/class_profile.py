import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
class profile:
    """DEPRECATED: use torch.profiler instead."""

    def __init__(self, enabled=True, *, use_cuda=False, record_shapes=False, with_flops=False, profile_memory=False, with_stack=False, with_modules=False):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        self.function_events = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        if self.use_cuda and (not torch.cuda.is_available()):
            warn('CUDA is not available, disabling CUDA profiling')
            self.use_cuda = False
        if self.use_cuda:
            self.profiler_kind = ProfilerState.CUDA
        else:
            self.profiler_kind = ProfilerState.CPU

    def config(self):
        return ProfilerConfig(self.profiler_kind, self.record_shapes, self.profile_memory, self.with_stack, self.with_flops, self.with_modules, torch._C._profiler._ExperimentalConfig())

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError('Profiler context manager is not reentrant')
        self.entered = True
        self._start_trace()
        return self

    def _start_trace(self):
        _enable_profiler_legacy(self.config())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()
        records = _disable_profiler_legacy()
        parsed_results = _parse_legacy_records(records)
        self.function_events = EventList(parsed_results, use_cuda=self.use_cuda, profile_memory=self.profile_memory, with_flops=self.with_flops)
        self.function_events._build_tree()
        return False

    def __repr__(self):
        if self.function_events is None:
            return '<unfinished profiler_legacy.profile>'
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return '<unfinished profile.profiler_legacy.profile>'
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, max_name_column_width=55, max_shapes_column_width=80, header=None, top_level_events_only=False):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.table(sort_by=sort_by, row_limit=row_limit, max_src_column_width=max_src_column_width, max_name_column_width=max_name_column_width, max_shapes_column_width=max_shapes_column_width, header=header, top_level_events_only=top_level_events_only)
    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.export_chrome_trace(path)
    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str='self_cpu_time_total'):
        self._check_finish()
        assert self.function_events is not None, 'Expected profiling results'
        assert self.with_stack, 'export_stacks() requires with_stack=True'
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        assert self.function_events is not None, 'Expected profiling results'
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)
    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        assert self.function_events is not None, 'Expected profiling results'
        return self.function_events.total_average()
    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Return CPU time as the sum of self times across all events."""
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.self_cpu_time_total