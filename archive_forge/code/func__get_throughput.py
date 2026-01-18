from vllm.logger import init_logger
from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY, disable_created_metrics
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
    return float(np.sum(tracked_stats) / (now - self.last_local_log))