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
def _create_evaluation_results(by_case_results):
    group_by_metric = dict()
    for case, case_result in by_case_results.items():
        for metric, evaluation_result in case_result.items():
            if metric not in group_by_metric:
                group_by_metric[metric] = list()
            group_by_metric[metric].append(evaluation_result)
    results = list()
    for metric, metric_results in group_by_metric.items():
        results.append(MetricEvaluationResult(metric_results))
    return EvaluationResults(results)