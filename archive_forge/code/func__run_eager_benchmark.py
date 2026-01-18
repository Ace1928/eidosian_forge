import time
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.platform import test
def _run_eager_benchmark(self, iterable, iters, warmup):
    """Benchmark the iterable in eager mode.

    Runs the iterable `iters` times. In each iteration, the benchmark measures
    the time it takes to go execute the iterable.

    Args:
      iterable: The tf op or tf.data Dataset to benchmark.
      iters: Number of times to repeat the timing.
      warmup: If true, warms up the session caches by running an untimed run.

    Returns:
      A float, representing the median time (with respect to `iters`)
      it takes for the iterable to be executed `iters` num of times.

    Raises:
      RuntimeError: When executed in graph mode.
    """
    deltas = []
    if not context.executing_eagerly():
        raise RuntimeError('Eager mode benchmarking is not supported in graph mode.')
    for _ in range(iters):
        if warmup:
            iterator = iter(iterable)
            next(iterator)
        iterator = iter(iterable)
        start = time.time()
        next(iterator)
        end = time.time()
        deltas.append(end - start)
    return np.median(deltas)