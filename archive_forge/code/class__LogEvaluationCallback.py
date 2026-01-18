from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
class _LogEvaluationCallback:
    """Internal log evaluation callable class."""

    def __init__(self, period: int=1, show_stdv: bool=True) -> None:
        self.order = 10
        self.before_iteration = False
        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env: CallbackEnv) -> None:
        if self.period > 0 and env.evaluation_result_list and ((env.iteration + 1) % self.period == 0):
            result = '\t'.join([_format_eval_result(x, self.show_stdv) for x in env.evaluation_result_list])
            _log_info(f'[{env.iteration + 1}]\t{result}')