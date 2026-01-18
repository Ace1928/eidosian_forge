import os
import subprocess
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Set
import numpy as np
import optuna
import torch
import transformers
from datasets import Dataset
from tqdm import trange
from . import version as optimum_version
from .utils.preprocessing import (
from .utils.runs import RunConfig, cpu_info_command
class TimeBenchmark:

    def __init__(self, model, batch_size: int, input_length: int, model_input_names: Set[str], warmup_runs: int, duration: float):
        self.batch_size = batch_size
        self.input_length = input_length
        self.model = model
        self.warmup_runs = warmup_runs
        self.benchmark_duration = duration
        self.latencies = []
        self.throughput = float('-inf')
        self.model_input_names = model_input_names

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @contextmanager
    def track(self):
        start = perf_counter_ns()
        yield
        end = perf_counter_ns()
        self.latencies.append(end - start)
        print(f'Tracked function took: {end - start}ns ({(end - start) / 1000000.0:.3f}ms)')

    def finalize(self, duration_ns: int):
        self.throughput = round(len(self.latencies) / duration_ns * SEC_TO_NS_SCALE, 2)

    def to_dict(self):
        benchmarks_stats = {'nb_forwards': len(self.latencies), 'throughput': self.throughput, 'latency_mean': ns_to_ms(np.mean(self.latencies)), 'latency_std': ns_to_ms(np.std(self.latencies)), 'latency_50': ns_to_ms(np.quantile(self.latencies, 0.5)), 'latency_90': ns_to_ms(np.quantile(self.latencies, 0.9)), 'latency_95': ns_to_ms(np.quantile(self.latencies, 0.95)), 'latency_99': ns_to_ms(np.quantile(self.latencies, 0.99)), 'latency_999': ns_to_ms(np.quantile(self.latencies, 0.999))}
        return benchmarks_stats

    def execute(self):
        inputs = {}
        checked_inputs = {'input_ids', 'attention_mask', 'token_type_ids', 'pixel_values'}
        if 'input_ids' in self.model_input_names:
            inputs['input_ids'] = torch.randint(high=1000, size=(self.batch_size, self.input_length))
        if 'attention_mask' in self.model_input_names:
            inputs['attention_mask'] = torch.ones(self.batch_size, self.input_length, dtype=torch.int64)
        if 'token_type_ids' in self.model_input_names:
            inputs['token_type_ids'] = torch.ones(self.batch_size, self.input_length, dtype=torch.int64)
        if 'pixel_values' in self.model_input_names:
            inputs['pixel_values'] = torch.rand(self.batch_size, 3, self.model.config.image_size, self.model.config.image_size, dtype=torch.float32)
        if np.any([k not in checked_inputs for k in self.model_input_names]):
            raise NotImplementedError(f'At least an input in {self.model_input_names} has no dummy generation for time benchmark.')
        for _ in trange(self.warmup_runs, desc='Warming up'):
            self.model.forward(**inputs)
        if self.benchmark_duration != 0:
            benchmark_duration_ns = self.benchmark_duration * SEC_TO_NS_SCALE
            print(f'Running time tracking in {self.benchmark_duration:.1f}s.')
            while sum(self.latencies) < benchmark_duration_ns:
                with self.track():
                    self.model.forward(**inputs)
            self.finalize(benchmark_duration_ns)
            return self.to_dict()
        else:
            benchmarks_stats = {'nb_forwards': 0, 'throughput': -1, 'latency_mean': -1}
            return benchmarks_stats