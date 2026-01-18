from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
def _gt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
    return curr_score > best_score + delta