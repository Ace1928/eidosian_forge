from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
def _compute_metrics(self, metrics, grouped_by_case_models, learn_folds, skipped_folds, rest_folds):
    metric_calcers = {}
    for case, case_models in grouped_by_case_models.items():
        metric_calcers[case] = list()
        for case_model in case_models:
            metric_calcer = case_model.create_metrics_calcer(metrics, eval_step=self._eval_step, thread_count=self._thread_count)
            metric_calcers[case].append(metric_calcer)
            if self._metric_descriptions is None:
                self._init_case_results(metric_calcer.metric_descriptions())
            elif self._metric_descriptions != metric_calcer.metric_descriptions():
                raise CatBoostError('Error: metric names should be consistent')
    for file_num, fold_file in enumerate(learn_folds + skipped_folds + rest_folds):
        pool = FoldModelsHandler._create_pool(fold_file, self._thread_count)
        for case, case_models in grouped_by_case_models.items():
            calcers = metric_calcers[case]
            for model_num, model in enumerate(case_models):
                if file_num != model_num:
                    calcers[model_num].add(pool)
    for case, case_models in grouped_by_case_models.items():
        calcers = metric_calcers[case]
        case_results = self._case_results[case]
        for calcer, model in zip(calcers, case_models):
            scores = calcer.eval_metrics()
            for metric in self._metric_descriptions:
                case_results[metric]._add(model, scores.get_result(metric))