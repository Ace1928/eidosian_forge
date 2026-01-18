import json
import logging
import os
from dataclasses import dataclass
from typing import List
import ray
from ray.train._internal.utils import get_address_and_port
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.util import PublicAPI
def _setup_tensorflow_environment(worker_addresses: List[str], index: int):
    """Set up distributed Tensorflow training information.

    This function should be called on each worker.

    Args:
        worker_addresses: Addresses of all the workers.
        index: Index (i.e. world rank) of the current worker.
    """
    tf_config = {'cluster': {'worker': worker_addresses}, 'task': {'type': 'worker', 'index': index}}
    os.environ['TF_CONFIG'] = json.dumps(tf_config)