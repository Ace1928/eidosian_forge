from vllm.logger import init_logger
from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, disable_created_metrics
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float
    num_running: int
    num_waiting: int
    num_swapped: int
    gpu_cache_usage: float
    cpu_cache_usage: float
    num_prompt_tokens: int
    num_generation_tokens: int
    time_to_first_tokens: List[float]
    time_per_output_tokens: List[float]
    time_e2e_requests: List[float]