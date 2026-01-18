from typing import Any, Callable, Dict, List, Optional
import optuna
from optuna.study import Study
from triad import SerializableRLock
from tune import (
from tune._utils.math import _IGNORABLE_ERROR, uniform_to_discrete, uniform_to_integers
from tune.concepts.logger import make_logger
from tune.concepts.space import TuningParametersTemplate
class OptunaLocalOptimizer(NonIterativeObjectiveLocalOptimizer):

    def __init__(self, max_iter: int, create_study: Optional[Callable[[], Study]]=None):
        self._max_iter = max_iter
        self._create_study = create_study or optuna.create_study

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial, logger: Any) -> TrialReport:
        template = trial.params
        if template.empty:
            tmp = NonIterativeObjectiveLocalOptimizer()
            return tmp.run(func, trial, logger=logger)
        lock = SerializableRLock()
        best_report: List[TrialReport] = []
        with make_logger(logger) as p_logger:
            with p_logger.create_child(name=trial.trial_id[:5] + '-' + p_logger.unique_id, description=repr(trial)) as c_logger:

                def obj(otrial: optuna.trial.Trial) -> float:
                    with c_logger.create_child(is_step=True) as s_logger:
                        params = template.fill_dict(_convert(otrial, template))
                        report = func.safe_run(trial.with_params(params))
                        with lock:
                            if len(best_report) == 0:
                                best_report.append(report)
                            elif report.sort_metric < best_report[0].sort_metric:
                                best_report[0] = report
                            s_logger.log_report(best_report[0])
                        return report.sort_metric
                study = self._create_study()
                study.optimize(obj, n_trials=self._max_iter)
                assert 1 == len(best_report)
                report = best_report[0]
                c_logger.log_params(report.trial.params.simple_value)
                c_logger.log_metrics({'OBJECTIVE_METRIC': report.metric})
                nm = {k: v for k, v in report.metadata.items() if isinstance(v, (int, float))}
                c_logger.log_metrics(nm)
                c_logger.log_metadata(report.metadata)
                return report