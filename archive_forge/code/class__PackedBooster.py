import copy
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
from .compat import SKLEARN_INSTALLED, DataFrame, XGBStratifiedKFold
from .core import (
class _PackedBooster:

    def __init__(self, cvfolds: _CVFolds) -> None:
        self.cvfolds = cvfolds

    def update(self, iteration: int, obj: Optional[Objective]) -> None:
        """Iterate through folds for update"""
        for fold in self.cvfolds:
            fold.update(iteration, obj)

    def eval(self, iteration: int, feval: Optional[Metric], output_margin: bool) -> List[str]:
        """Iterate through folds for eval"""
        result = [f.eval(iteration, feval, output_margin) for f in self.cvfolds]
        return result

    def set_attr(self, **kwargs: Optional[Any]) -> Any:
        """Iterate through folds for setting attributes"""
        for f in self.cvfolds:
            f.bst.set_attr(**kwargs)

    def attr(self, key: str) -> Optional[str]:
        """Redirect to booster attr."""
        return self.cvfolds[0].bst.attr(key)

    def set_param(self, params: Union[Dict, Iterable[Tuple[str, Any]], str], value: Optional[str]=None) -> None:
        """Iterate through folds for set_param"""
        for f in self.cvfolds:
            f.bst.set_param(params, value)

    def num_boosted_rounds(self) -> int:
        """Number of boosted rounds."""
        return self.cvfolds[0].num_boosted_rounds()

    @property
    def best_iteration(self) -> int:
        """Get best_iteration"""
        return int(cast(int, self.cvfolds[0].bst.attr('best_iteration')))

    @best_iteration.setter
    def best_iteration(self, iteration: int) -> None:
        """Get best_iteration"""
        self.set_attr(best_iteration=iteration)

    @property
    def best_score(self) -> float:
        """Get best_score."""
        return float(cast(float, self.cvfolds[0].bst.attr('best_score')))

    @best_score.setter
    def best_score(self, score: float) -> None:
        self.set_attr(best_score=score)