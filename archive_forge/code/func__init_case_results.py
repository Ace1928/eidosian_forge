from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
def _init_case_results(self, metric_descriptions):
    self._metric_descriptions = metric_descriptions
    for case in self._cases:
        case_result = self._case_results[case]
        for metric_description in self._metric_descriptions:
            case_result[metric_description] = CaseEvaluationResult(case, metric_description, eval_step=self._eval_step)