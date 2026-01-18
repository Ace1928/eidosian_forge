from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
def _final_iteration_check(self, env: CallbackEnv, eval_name_splitted: List[str], i: int) -> None:
    if env.iteration == env.end_iteration - 1:
        if self.verbose:
            best_score_str = '\t'.join([_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]])
            _log_info(f'Did not meet early stopping. Best iteration is:\n[{self.best_iter[i] + 1}]\t{best_score_str}')
            if self.first_metric_only:
                _log_info(f'Evaluated only: {eval_name_splitted[-1]}')
        raise EarlyStopException(self.best_iter[i], self.best_score_list[i])