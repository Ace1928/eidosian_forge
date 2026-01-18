from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
class RemoteTrialJudge(TrialJudge):

    def __init__(self, entrypoint: Callable[[str, Dict[str, Any]], Any]):
        super().__init__()
        self._entrypoint = entrypoint
        self._report: Optional[TrialReport] = None

    @property
    def report(self) -> Optional[TrialReport]:
        return self._report

    def can_accept(self, trial: Trial) -> bool:
        return self._entrypoint('can_accept', dict(trial=trial))

    def judge(self, report: TrialReport) -> TrialDecision:
        self._report = report
        return self._entrypoint('judge', dict(report=report))

    def get_budget(self, trial: Trial, rung: int) -> float:
        return self._entrypoint('get_budget', dict(trial=trial, rung=rung))