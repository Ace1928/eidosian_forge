import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
def _update_rounds(self, score: _Score, name: str, metric: str, model: _Model, epoch: int) -> bool:

    def get_s(value: _Score) -> float:
        """get score if it's cross validation history."""
        return value[0] if isinstance(value, tuple) else value

    def maximize(new: _Score, best: _Score) -> bool:
        """New score should be greater than the old one."""
        return numpy.greater(get_s(new) - self._min_delta, get_s(best))

    def minimize(new: _Score, best: _Score) -> bool:
        """New score should be lesser than the old one."""
        return numpy.greater(get_s(best) - self._min_delta, get_s(new))
    if self.maximize is None:
        maximize_metrics = ('auc', 'aucpr', 'pre', 'pre@', 'map', 'ndcg', 'auc@', 'aucpr@', 'map@', 'ndcg@')
        if metric != 'mape' and any((metric.startswith(x) for x in maximize_metrics)):
            self.maximize = True
        else:
            self.maximize = False
    if self.maximize:
        improve_op = maximize
    else:
        improve_op = minimize
    if not self.stopping_history:
        self.current_rounds = 0
        self.stopping_history[name] = {}
        self.stopping_history[name][metric] = cast(_ScoreList, [score])
        self.best_scores[name] = {}
        self.best_scores[name][metric] = [score]
        model.set_attr(best_score=str(score), best_iteration=str(epoch))
    elif not improve_op(score, self.best_scores[name][metric][-1]):
        self.stopping_history[name][metric].append(score)
        self.current_rounds += 1
    else:
        self.stopping_history[name][metric].append(score)
        self.best_scores[name][metric].append(score)
        record = self.stopping_history[name][metric][-1]
        model.set_attr(best_score=str(record), best_iteration=str(epoch))
        self.current_rounds = 0
    if self.current_rounds >= self.rounds:
        return True
    return False