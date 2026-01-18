from vllm.logger import init_logger
from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, disable_created_metrics
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
def _log_prometheus(self, stats: Stats) -> None:
    self.metrics.gauge_scheduler_running.labels(**self.labels).set(stats.num_running)
    self.metrics.gauge_scheduler_swapped.labels(**self.labels).set(stats.num_swapped)
    self.metrics.gauge_scheduler_waiting.labels(**self.labels).set(stats.num_waiting)
    self.metrics.gauge_gpu_cache_usage.labels(**self.labels).set(stats.gpu_cache_usage)
    self.metrics.gauge_cpu_cache_usage.labels(**self.labels).set(stats.cpu_cache_usage)
    self.metrics.counter_prompt_tokens.labels(**self.labels).inc(stats.num_prompt_tokens)
    self.metrics.counter_generation_tokens.labels(**self.labels).inc(stats.num_generation_tokens)
    for ttft in stats.time_to_first_tokens:
        self.metrics.histogram_time_to_first_token.labels(**self.labels).observe(ttft)
    for tpot in stats.time_per_output_tokens:
        self.metrics.histogram_time_per_output_token.labels(**self.labels).observe(tpot)
    for e2e in stats.time_e2e_requests:
        self.metrics.histogram_e2e_request_latency.labels(**self.labels).observe(e2e)