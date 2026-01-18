from __future__ import print_function
import os
from copy import copy
from enum import Enum
from .. import CatBoostError, CatBoost
from .evaluation_result import EvaluationResults, MetricEvaluationResult
from ._fold_models_handler import FoldModelsHandler
from ._readers import _SimpleStreamingFileReader
from ._splitter import _Splitter
from .execution_case import ExecutionCase
from .factor_utils import LabelMode, FactorUtils
@staticmethod
def _create_eval_feature_cases(params, features_to_eval, eval_type, label_mode):
    if len(features_to_eval) == 0:
        raise CatBoostError('Provide at least one feature to evaluation')
    test_cases = list()
    baseline_case = ExecutionCase(params, ignored_features=list(features_to_eval), label=FactorUtils.create_label(features_to_eval, features_to_eval, label_mode=label_mode))
    if eval_type == EvalType.All or eval_type == EvalType.SeqAddAndAll or len(features_to_eval) == 1:
        test_cases.append(ExecutionCase(params, ignored_features=[], label=FactorUtils.create_label(features_to_eval, [], label_mode=label_mode)))
    elif eval_type == EvalType.SeqRem:
        for feature_num in features_to_eval:
            test_cases.append(ExecutionCase(params, ignored_features=[feature_num], label=FactorUtils.create_label(features_to_eval, [feature_num], label_mode=label_mode)))
    elif eval_type == EvalType.SeqAdd or eval_type == EvalType.SeqAddAndAll:
        for feature_num in features_to_eval:
            cur_features = copy(features_to_eval)
            cur_features.remove(feature_num)
            test_cases.append(ExecutionCase(params, label=FactorUtils.create_label(features_to_eval, cur_features, label_mode=label_mode), ignored_features=list(cur_features)))
    elif eval_type != EvalType.All:
        raise AttributeError("Don't support {} mode.", eval_type.value)
    return (baseline_case, test_cases)