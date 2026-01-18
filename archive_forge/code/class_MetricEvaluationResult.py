import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
class MetricEvaluationResult:
    """
        Evaluation result for one metric.
        Stores all ExecutionCases with specified metric scores
        Computes human-friendly tables with results and some plots
    """

    def __init__(self, case_results):
        if len(case_results) <= 1:
            raise CatBoostError('Need at least 2 case results, got {} '.format(len(case_results)))
        self._case_results = dict()
        self._case_comparisons = dict()
        self._cases = [case_result.get_case() for case_result in case_results]
        for case_result in case_results:
            case = case_result.get_case()
            self._case_results[case] = case_result
        self._metric_description = case_results[0].get_metric_description()
        self._baseline_case = case_results[0].get_case()
        self._score_config = ScoreConfig()
        for case, case_result in self._case_results.items():
            if case_result.get_metric_description() != self._metric_description:
                raise CatBoostError('Metric names should be equal for all case results')
            if case_result.get_fold_ids() != self.get_fold_ids():
                raise CatBoostError('Case results should be computed on the same folds')
            if case_result.get_eval_step() != self.get_eval_step():
                raise CatBoostError('Eval steps should be equal for different cases')

    def __clear_comparisons(self):
        self._case_comparisons = dict()

    def _change_score_config(self, config):
        if config is not None:
            if isinstance(config, ScoreType):
                if config == ScoreType.Abs:
                    config = ScoreConfig.abs_score()
                elif config == ScoreType.Rel:
                    config = ScoreConfig.rel_score()
                else:
                    raise CatBoostError('Unknown scoreType {}'.format(config))
            if self._score_config != config:
                self._score_config = config
                self.__clear_comparisons()

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

    def get_baseline_case(self):
        """

        :return: ExecutionCases used as a baseline (with everything else is compared)
        """
        return self._baseline_case

    def get_cases(self):
        """

        :return: Cases which are compared
        """
        return self._cases

    def get_metric_description(self):
        """

        :return: Metric for which results were calculated
        """
        return self._metric_description

    def get_baseline_comparison(self, score_config=None):
        """
        Method to get human-friendly table with model comparisons.

        Returns baseline vs all other computed cases result
        :param score_config: Config to present human-friendly score, optional. Instance of ScoreConfig
        :return: pandas DataFrame. Each row is related to one ExecutionCase.
        Each row describes how better (or worse) this case is compared to baseline.
        """
        case = self._baseline_case
        return self.get_case_comparison(case, score_config)

    def get_case_comparison(self, case, score_config=None):
        """
        Method to get human-friendly table with model comparisons.
        Same as get_baseline_comparison(), but with other non-baseline case specified as baseline

        :param case: use specified case as baseline
        :param score_config:
        :return: pandas DataFrame. Each row is related to one ExecutionCase.
        Each row describes how better (or worse) this case is compared to baseline.
        """
        self._change_score_config(score_config)
        if case not in self._case_comparisons:
            self._case_comparisons[case] = self._compute_case_result_table(case)
        return self._case_comparisons[case]

    def change_baseline_case(self, case):
        """

        :param case: new baseline case
        :return:
        """
        if case not in self._case_results:
            raise CatBoostError("Case {} is unknown. Can't use it as baseline".format(case))
        self._baseline_case = case

    def get_case_result(self, case):
        """

        :param case:
        :return: CaseEvaluationResult. Scores and other information about single execution case
        """
        return self._case_results[case]

    def get_fold_ids(self):
        """

        :return: Folds ids which we used for computing this evaluation result
        """
        return self._case_results[self._baseline_case].get_fold_ids()

    def get_eval_step(self):
        return self._case_results[self._baseline_case].get_eval_step()

    def create_fold_learning_curves(self, fold, offset=None):
        """

        :param fold: FoldId to plot
        :param offset: first iteration to plot
        :return: plotly figure for all cases on specified fold
        """
        import plotly.graph_objs as go
        traces = []
        for case in self.get_cases():
            case_result = self.get_case_result(case)
            scores_curve = case_result.get_fold_curve(fold)
            if offset is not None:
                first_idx = offset
            else:
                first_idx = int(len(scores_curve) * 0.1)
            traces.append(go.Scatter(x=[i * int(case_result.get_eval_step()) for i in range(first_idx, len(scores_curve))], y=scores_curve[first_idx:], mode='lines', name='Case {}'.format(case)))
        layout = go.Layout(title='Learning curves for metric {} on fold #{}'.format(self._metric_description, fold), hovermode='closest', xaxis=dict(title='Iteration', ticklen=5, zeroline=False, gridwidth=2), yaxis=dict(title='Metric', ticklen=5, gridwidth=2), showlegend=True)
        fig = go.Figure(data=traces, layout=layout)
        return fig

    def __eq__(self, other):
        return self._case_results == other._case_results and self._case_comparisons == other._case_comparisons and (self._cases == other._cases)