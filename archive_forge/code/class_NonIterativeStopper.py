from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
class NonIterativeStopper(TrialJudge):

    def __init__(self, log_best_only: bool=False):
        super().__init__()
        self._stopper_updated = False
        self._lock = SerializableRLock()
        self._log_best_only = log_best_only
        self._logs: Dict[str, TrialReportCollection] = {}

    @property
    def updated(self) -> bool:
        return self._stopper_updated

    def should_stop(self, trial: Trial) -> bool:
        return False

    def on_report(self, report: TrialReport) -> bool:
        self._stopper_updated = True
        self.monitor.on_report(report)
        with self._lock:
            key = str(report.trial.keys)
            if key not in self._logs:
                self._logs[key] = TrialReportCollection(self._log_best_only)
            return self._logs[key].on_report(report)

    def can_accept(self, trial: Trial) -> bool:
        return not self.should_stop(trial)

    def judge(self, report: TrialReport) -> TrialDecision:
        self.on_report(report)
        return TrialDecision(report, 0.0, False)

    def get_reports(self, trial: Trial) -> List[TrialReport]:
        with self._lock:
            key = str(trial.keys)
            if key not in self._logs:
                return []
            v = self._logs[key]
        return v.reports

    def __and__(self, other: 'NonIterativeStopper') -> 'NonIterativeStopperCombiner':
        return NonIterativeStopperCombiner(self, other, is_and=True)

    def __or__(self, other: 'NonIterativeStopper') -> 'NonIterativeStopperCombiner':
        return NonIterativeStopperCombiner(self, other, is_and=False)