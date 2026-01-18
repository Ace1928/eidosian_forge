import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def _parse_characterized_angles(metrics: 'cirq_google.Calibration', super_name: str) -> Dict[Tuple[cirq.Qid, cirq.Qid], Dict[str, float]]:
    """Parses characterized angles from Metric protos.

    Args:
        metrics: The metrics from a CalibrationResult
        super_name: The metric name prefix. We extract angle names as "{super_name}_{angle_name}".
    """
    records: Dict[Tuple[cirq.Qid, cirq.Qid], Dict[str, float]] = collections.defaultdict(dict)
    for metric_name in metrics.keys():
        ma = re.match(f'{super_name}_(\\w+)$', metric_name)
        if ma is None:
            continue
        angle_name = ma.group(1)
        for (qa, qb), (value,) in metrics[metric_name].items():
            qa = cast(cirq.GridQubit, qa)
            qb = cast(cirq.GridQubit, qb)
            value = float(value)
            records[qa, qb][angle_name] = value
    return dict(records)