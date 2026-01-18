from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
def _reset_storages(self) -> None:
    self.best_score: List[float] = []
    self.best_iter: List[int] = []
    self.best_score_list: List[_ListOfEvalResultTuples] = []
    self.cmp_op: List[Callable[[float, float], bool]] = []
    self.first_metric = ''