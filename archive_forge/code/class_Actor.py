import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
@ray.remote(num_cpus=0)
class Actor:

    def small_value(self):
        return b'ok'

    def small_value_arg(self, x):
        return b'ok'

    def small_value_batch(self, n):
        ray.get([small_value.remote() for _ in range(n)])