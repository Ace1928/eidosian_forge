import math
import numbers
import os
import re
import sys
import time
import types
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _run_benchmarks(regex):
    """Run benchmarks that match regex `regex`.

  This function goes through the global benchmark registry, and matches
  benchmark class and method names of the form
  `module.name.BenchmarkClass.benchmarkMethod` to the given regex.
  If a method matches, it is run.

  Args:
    regex: The string regular expression to match Benchmark classes against.

  Raises:
    ValueError: If no benchmarks were selected by the input regex.
  """
    registry = list(GLOBAL_BENCHMARK_REGISTRY)
    selected_benchmarks = []
    for benchmark in registry:
        benchmark_name = '%s.%s' % (benchmark.__module__, benchmark.__name__)
        attrs = dir(benchmark)
        benchmark_instance = None
        for attr in attrs:
            if not attr.startswith('benchmark'):
                continue
            candidate_benchmark_fn = getattr(benchmark, attr)
            if not callable(candidate_benchmark_fn):
                continue
            full_benchmark_name = '%s.%s' % (benchmark_name, attr)
            if regex == 'all' or re.search(regex, full_benchmark_name):
                selected_benchmarks.append(full_benchmark_name)
                benchmark_instance = benchmark_instance or benchmark()
                instance_benchmark_fn = getattr(benchmark_instance, attr)
                instance_benchmark_fn()
    if not selected_benchmarks:
        raise ValueError("No benchmarks matched the pattern: '{}'".format(regex))