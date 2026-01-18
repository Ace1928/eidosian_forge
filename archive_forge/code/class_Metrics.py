from vllm.logger import init_logger
from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, disable_created_metrics
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
class Metrics:

    def __init__(self, labelnames: List[str]):
        for collector in list(REGISTRY._collector_to_names):
            if hasattr(collector, '_name') and 'vllm' in collector._name:
                REGISTRY.unregister(collector)
        self.info_cache_config = Info(name='vllm:cache_config', documentation='information of cache_config')
        self.gauge_scheduler_running = Gauge(name='vllm:num_requests_running', documentation='Number of requests currently running on GPU.', labelnames=labelnames)
        self.gauge_scheduler_swapped = Gauge(name='vllm:num_requests_swapped', documentation='Number of requests swapped to CPU.', labelnames=labelnames)
        self.gauge_scheduler_waiting = Gauge(name='vllm:num_requests_waiting', documentation='Number of requests waiting to be processed.', labelnames=labelnames)
        self.gauge_gpu_cache_usage = Gauge(name='vllm:gpu_cache_usage_perc', documentation='GPU KV-cache usage. 1 means 100 percent usage.', labelnames=labelnames)
        self.gauge_cpu_cache_usage = Gauge(name='vllm:cpu_cache_usage_perc', documentation='CPU KV-cache usage. 1 means 100 percent usage.', labelnames=labelnames)
        self.counter_prompt_tokens = Counter(name='vllm:prompt_tokens_total', documentation='Number of prefill tokens processed.', labelnames=labelnames)
        self.counter_generation_tokens = Counter(name='vllm:generation_tokens_total', documentation='Number of generation tokens processed.', labelnames=labelnames)
        self.histogram_time_to_first_token = Histogram(name='vllm:time_to_first_token_seconds', documentation='Histogram of time to first token in seconds.', labelnames=labelnames, buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])
        self.histogram_time_per_output_token = Histogram(name='vllm:time_per_output_token_seconds', documentation='Histogram of time per output token in seconds.', labelnames=labelnames, buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5])
        self.histogram_e2e_request_latency = Histogram(name='vllm:e2e_request_latency_seconds', documentation='Histogram of end to end request latency in seconds.', labelnames=labelnames, buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.gauge_avg_prompt_throughput = Gauge(name='vllm:avg_prompt_throughput_toks_per_s', documentation='Average prefill throughput in tokens/s.', labelnames=labelnames)
        self.gauge_avg_generation_throughput = Gauge(name='vllm:avg_generation_throughput_toks_per_s', documentation='Average generation throughput in tokens/s.', labelnames=labelnames)