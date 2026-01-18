import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
def _aggcv(rlist: List[str]) -> List[Tuple[str, float, float]]:
    """Aggregate cross-validation results."""
    cvmap: Dict[Tuple[int, str], List[float]] = {}
    idx = rlist[0].split()[0]
    for line in rlist:
        arr: List[str] = line.split()
        assert idx == arr[0]
        for metric_idx, it in enumerate(arr[1:]):
            if not isinstance(it, str):
                it = it.decode()
            k, v = it.split(':')
            if (metric_idx, k) not in cvmap:
                cvmap[metric_idx, k] = []
            cvmap[metric_idx, k].append(float(v))
    msg = idx
    results = []
    for (_, name), s in sorted(cvmap.items(), key=lambda x: x[0][0]):
        as_arr = numpy.array(s)
        if not isinstance(msg, str):
            msg = msg.decode()
        mean, std = (numpy.mean(as_arr), numpy.std(as_arr))
        results.extend([(name, mean, std)])
    return results