import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
def _fmt_metric(self, data: str, metric: str, score: float, std: Optional[float]) -> str:
    if std is not None and self.show_stdv:
        msg = f'\t{data + '-' + metric}:{score:.5f}+{std:.5f}'
    else:
        msg = f'\t{data + '-' + metric}:{score:.5f}'
    return msg