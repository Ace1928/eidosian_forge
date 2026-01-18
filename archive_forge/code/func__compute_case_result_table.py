import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def _compute_case_result_table(self, baseline_case):
    result = pd.DataFrame()
    baseline_scores = self._case_results[baseline_case].get_best_metrics()
    baseline_iters = self._case_results[baseline_case].get_best_iterations()
    for case, case_result in self._case_results.items():
        if case != baseline_case:
            test_scores = case_result.get_best_metrics()
            pvalue = calc_wilcoxon_test(baseline_scores, test_scores)
            diff = baseline_scores - test_scores
            if self._score_config.type == ScoreType.Rel:
                diff = diff / baseline_scores.abs()
            if self._metric_description.is_max_optimal():
                diff = -diff
            mean_diff = diff.mean()
            left_quantile, right_quantile = calc_bootstrap_ci_for_mean(diff, self._score_config.interval_level)
            case_name = str(case)
            result.at[case_name, 'PValue'] = pvalue
            result.at[case_name, 'Score'] = mean_diff * self._score_config.multiplier
            left_quantile_title = 'Quantile {}'.format(self._score_config.score_level / 2)
            right_quantile_title = 'Quantile {}'.format(1.0 - self._score_config.score_level / 2)
            result.at[case_name, left_quantile_title] = left_quantile * self._score_config.multiplier
            result.at[case_name, right_quantile_title] = right_quantile * self._score_config.multiplier
            decision = 'UNKNOWN'
            if pvalue < self._score_config.score_level:
                if mean_diff > 0:
                    decision = 'GOOD'
                elif mean_diff < 0:
                    decision = 'BAD'
            result.at[case_name, 'Decision'] = decision
            if self._score_config.overfit_overfit_iterations_info:
                test_iters = case_result.get_best_iterations()
                pvalue = calc_wilcoxon_test(baseline_iters, test_iters)
                result.at[case_name, 'Overfit iter diff'] = (test_iters - baseline_iters).mean()
                result.at[case_name, 'Overfit iter pValue'] = pvalue
    return result.sort_values(by=['Score'], ascending=self._metric_description.is_max_optimal())