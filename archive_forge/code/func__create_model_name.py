from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
@staticmethod
def _create_model_name(model_case, fold):
    import uuid
    id_str = str(uuid.uuid1()).replace('-', '_')
    model_name = 'model_{_name}_fold_{_fold}_{_uuid}.bin'.format(_name=model_case.get_label(), _fold=fold, _uuid=id_str)
    return model_name