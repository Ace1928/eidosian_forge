from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import queue
from socketserver import ThreadingMixIn
import threading
import time
import traceback
from typing import List
import ray.cloudpickle as pickle
from ray.rllib.env.policy_client import (
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SamplerInput
from ray.rllib.utils.typing import SampleBatchType
class MetricsDummySampler(SamplerInput):
    """This sampler only maintains a queue to get metrics from."""

    def __init__(self, metrics_queue):
        """Initializes a MetricsDummySampler instance.

                    Args:
                        metrics_queue: A queue of metrics
                    """
        self.metrics_queue = metrics_queue

    def get_data(self) -> SampleBatchType:
        raise NotImplementedError

    def get_extra_batches(self) -> List[SampleBatchType]:
        raise NotImplementedError

    def get_metrics(self) -> List[RolloutMetrics]:
        """Returns metrics computed on a policy client rollout worker."""
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed