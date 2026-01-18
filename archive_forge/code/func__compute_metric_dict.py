from collections import abc, defaultdict
import datetime
from itertools import cycle
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import google.protobuf.json_format as json_format
import cirq
from cirq_google.api import v2
def _compute_metric_dict(self, metrics: v2.metrics_pb2.MetricsSnapshot) -> ALL_METRICS:
    results: ALL_METRICS = defaultdict(dict)
    for metric in metrics:
        name = metric.name
        flat_values = [getattr(v, v.WhichOneof('val')) for v in metric.values]
        if metric.targets:
            qubits = tuple((self.str_to_key(t) for t in metric.targets))
            results[name][qubits] = flat_values
        else:
            assert len(results[name]) == 0, f'Only one metric of a given name can have no targets. Found multiple for key {name}'
            results[name][()] = flat_values
    return results