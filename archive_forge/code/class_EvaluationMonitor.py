import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
class EvaluationMonitor(TrainingCallback):
    """Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    rank :
        Which worker should be used for printing the result.
    period :
        How many epoches between printing.
    show_stdv :
        Used in cv to show standard deviation.  Users should not specify it.
    """

    def __init__(self, rank: int=0, period: int=1, show_stdv: bool=False) -> None:
        self.printer_rank = rank
        self.show_stdv = show_stdv
        self.period = period
        assert period > 0
        self._latest: Optional[str] = None
        super().__init__()

    def _fmt_metric(self, data: str, metric: str, score: float, std: Optional[float]) -> str:
        if std is not None and self.show_stdv:
            msg = f'\t{data + '-' + metric}:{score:.5f}+{std:.5f}'
        else:
            msg = f'\t{data + '-' + metric}:{score:.5f}'
        return msg

    def after_iteration(self, model: _Model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        if not evals_log:
            return False
        msg: str = f'[{epoch}]'
        if collective.get_rank() == self.printer_rank:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    stdv: Optional[float] = None
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                        stdv = log[-1][1]
                    else:
                        score = log[-1]
                    msg += self._fmt_metric(data, metric_name, score, stdv)
            msg += '\n'
            if epoch % self.period == 0 or self.period == 1:
                collective.communicator_print(msg)
                self._latest = None
            else:
                self._latest = msg
        return False

    def after_training(self, model: _Model) -> _Model:
        if collective.get_rank() == self.printer_rank and self._latest is not None:
            collective.communicator_print(self._latest)
        return model