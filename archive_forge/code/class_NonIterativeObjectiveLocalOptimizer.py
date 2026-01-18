from typing import Any, Callable, Optional
from tune._utils import run_monitored_process
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.logger import make_logger
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
class NonIterativeObjectiveLocalOptimizer:

    @property
    def distributable(self) -> bool:
        return True

    def run(self, func: NonIterativeObjectiveFunc, trial: Trial, logger: Any) -> TrialReport:
        if logger is None:
            report = func.safe_run(trial)
        else:
            with make_logger(logger) as p_logger:
                with p_logger.create_child(name=trial.trial_id[:5] + '-' + p_logger.unique_id, description=repr(trial)) as c_logger:
                    report = func.safe_run(trial)
                    c_logger.log_report(report, log_params=True, extract_metrics=True, log_metadata=True)
        return report

    def run_monitored_process(self, func: NonIterativeObjectiveFunc, trial: Trial, stop_checker: Callable[[], bool], logger: Any, interval: Any=TUNE_STOPPER_DEFAULT_CHECK_INTERVAL) -> TrialReport:
        return run_monitored_process(self.run, [func, trial], {'logger': logger}, stop_checker=stop_checker, interval=interval)