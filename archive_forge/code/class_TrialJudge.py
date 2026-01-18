from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
class TrialJudge:

    def __init__(self, monitor: Optional['Monitor']=None):
        self.reset_monitor(monitor)

    @property
    def monitor(self) -> 'Monitor':
        assert self._trial_judge_monitor is not None
        return self._trial_judge_monitor

    def reset_monitor(self, monitor: Optional['Monitor']=None) -> None:
        self._trial_judge_monitor = monitor or Monitor()

    def can_accept(self, trial: Trial) -> bool:
        raise NotImplementedError

    def get_budget(self, trial: Trial, rung: int) -> float:
        raise NotImplementedError

    def judge(self, report: TrialReport) -> TrialDecision:
        raise NotImplementedError