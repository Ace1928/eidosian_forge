from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
@staticmethod
def _remove_models(list_models):
    get_eval_logger().debug('Remove models {}'.format(str(list_models)))
    for model in list_models:
        model.delete()