from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
def _parse_kineto_results(self, result: _ProfilerResult):
    trace_start_us = result.trace_start_us()
    mem_records = [[evt, False] for evt in result.events() if evt.name() == MEMORY_EVENT_NAME]
    oom_records = [evt for evt in result.events() if evt.name() == OUT_OF_MEMORY_EVENT_NAME]
    mem_records_acc = MemRecordsAcc(mem_records)

    def _cpu_memory_usage(mem_record):
        return mem_record.nbytes() if mem_record.device_type() in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP] else 0

    def _cuda_memory_usage(mem_record):
        return mem_record.nbytes() if mem_record.device_type() in [DeviceType.CUDA, DeviceType.HIP] else 0

    def _privateuse1_memory_usage(mem_record):
        return mem_record.nbytes() if mem_record.device_type() in [DeviceType.PrivateUse1] else 0
    function_events = []
    device_corr_map: Dict[int, List[FunctionEvent]] = {}
    max_evt_id = 0
    for kineto_event in result.events():
        if _filter_name(kineto_event.name()):
            continue
        rel_start_us = kineto_event.start_us() - trace_start_us
        rel_end_us = rel_start_us + kineto_event.duration_us()
        abs_end_us = kineto_event.start_us() + kineto_event.duration_us()
        cpu_memory_usage = 0
        cuda_memory_usage = 0
        privateuse1_memory_usage = 0
        if kineto_event.device_type() == DeviceType.CPU:
            for mem_record in mem_records_acc.in_interval(kineto_event.start_us(), abs_end_us):
                cpu_memory_usage += _cpu_memory_usage(mem_record[0])
                cuda_memory_usage += _cuda_memory_usage(mem_record[0])
                privateuse1_memory_usage += _privateuse1_memory_usage(mem_record[0])
                mem_record[1] = True
        is_async = kineto_event.is_async() or kineto_event.start_thread_id() != kineto_event.end_thread_id()
        fe = FunctionEvent(id=kineto_event.correlation_id(), name=_rewrite_name(name=kineto_event.name(), with_wildcard=True), trace_name=_rewrite_name(name=kineto_event.name(), with_wildcard=False), thread=kineto_event.start_thread_id(), start_us=rel_start_us, end_us=rel_end_us, fwd_thread=kineto_event.fwd_thread_id(), input_shapes=kineto_event.shapes(), concrete_inputs=kineto_event.concrete_inputs(), stack=[entry for entry in kineto_event.stack() if _filter_stack_entry(entry)], scope=kineto_event.scope(), use_device=self.use_device, cpu_memory_usage=cpu_memory_usage, cuda_memory_usage=cuda_memory_usage, privateuse1_memory_usage=privateuse1_memory_usage, is_async=is_async, sequence_nr=kineto_event.sequence_nr(), device_type=kineto_event.device_type(), device_index=kineto_event.device_index(), flops=kineto_event.flops())
        max_evt_id = max(max_evt_id, fe.id)
        if fe.device_type == DeviceType.CPU and (not fe.is_async):
            if self.use_device:
                privateuse1_time = kineto_event.privateuse1_elapsed_us()
                if privateuse1_time > 0:
                    fe.append_kernel(fe.name, fe.device_index, privateuse1_time)
                    fe.is_legacy = True
            else:
                cuda_time = kineto_event.cuda_elapsed_us()
                if cuda_time > 0:
                    fe.append_kernel(fe.name, fe.device_index, cuda_time)
                    fe.is_legacy = True
        function_events.append(fe)
        corr_id = kineto_event.linked_correlation_id()
        if corr_id > 0:
            if corr_id not in device_corr_map:
                device_corr_map[corr_id] = []
            device_corr_map[corr_id].append(fe)
    for fe in function_events:
        if fe.device_type == DeviceType.CPU and (not fe.is_async) and (fe.id in device_corr_map):
            for f_evt in device_corr_map[fe.id]:
                if f_evt.device_type == DeviceType.CUDA:
                    fe.append_kernel(f_evt.name, f_evt.device_index, f_evt.time_range.end - f_evt.time_range.start)
                elif f_evt.device_type == DeviceType.CPU:
                    f_evt.thread = fe.thread

    def createFunctionEventForMemoryEvents(evt):
        rel_start_us = evt.start_us() - trace_start_us
        fe = FunctionEvent(id=max_evt_id, name=evt.name(), trace_name=None, thread=evt.start_thread_id(), start_us=rel_start_us, end_us=rel_start_us, fwd_thread=evt.start_thread_id(), input_shapes=[], stack=[], scope=0, use_device=self.use_device, cpu_memory_usage=_cpu_memory_usage(evt), cuda_memory_usage=_cuda_memory_usage(evt), privateuse1_memory_usage=_privateuse1_memory_usage(evt), is_async=False, sequence_nr=-1, device_type=DeviceType.CPU, device_index=0)
        return fe
    for mem_record in mem_records:
        if not mem_record[1]:
            max_evt_id += 1
            fe = createFunctionEventForMemoryEvents(mem_record[0])
            function_events.append(fe)
    for oom_record in oom_records:
        max_evt_id += 1
        fe = createFunctionEventForMemoryEvents(oom_record)
        function_events.append(fe)
    function_events.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])
    return function_events