from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
class EarlyStopException(Exception):
    """Exception of early stopping.

    Raise this from a callback passed in via keyword argument ``callbacks``
    in ``cv()`` or ``train()`` to trigger early stopping.
    """

    def __init__(self, best_iteration: int, best_score: _ListOfEvalResultTuples) -> None:
        """Create early stopping exception.

        Parameters
        ----------
        best_iteration : int
            The best iteration stopped.
            0-based... pass ``best_iteration=2`` to indicate that the third iteration was the best one.
        best_score : list of (eval_name, metric_name, eval_result, is_higher_better) tuple or (eval_name, metric_name, eval_result, is_higher_better, stdv) tuple
            Scores for each metric, on each validation set, as of the best iteration.
        """
        super().__init__()
        self.best_iteration = best_iteration
        self.best_score = best_score