import copy
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
from .compat import SKLEARN_INSTALLED, DataFrame, XGBStratifiedKFold
from .core import (
class CVPack:
    """ "Auxiliary datastruct to hold one fold of CV."""

    def __init__(self, dtrain: DMatrix, dtest: DMatrix, param: Optional[Union[Dict, List]]) -> None:
        """ "Initialize the CVPack"""
        self.dtrain = dtrain
        self.dtest = dtest
        self.watchlist = [(dtrain, 'train'), (dtest, 'test')]
        self.bst = Booster(param, [dtrain, dtest])

    def __getattr__(self, name: str) -> Callable:

        def _inner(*args: Any, **kwargs: Any) -> Any:
            return getattr(self.bst, name)(*args, **kwargs)
        return _inner

    def update(self, iteration: int, fobj: Optional[Objective]) -> None:
        """ "Update the boosters for one iteration"""
        self.bst.update(self.dtrain, iteration, fobj)

    def eval(self, iteration: int, feval: Optional[Metric], output_margin: bool) -> str:
        """ "Evaluate the CVPack for one iteration."""
        return self.bst.eval_set(self.watchlist, iteration, feval, output_margin)