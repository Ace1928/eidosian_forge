import os
import tempfile
from typing import Callable, List, Optional
from uuid import uuid4
import cloudpickle
from fs.base import FS as FSBase
from triad import FileSystem
from tune.concepts.checkpoint import Checkpoint
from tune.concepts.flow import Monitor, Trial, TrialDecision, TrialJudge, TrialReport
class IterativeObjectiveFunc:

    def __init__(self):
        self._rung = 0
        self._current_trial: Optional[Trial] = None

    def copy(self) -> 'IterativeObjectiveFunc':
        raise NotImplementedError

    @property
    def current_trial(self) -> Trial:
        assert self._current_trial is not None
        return self._current_trial

    @property
    def rung(self) -> int:
        return self._rung

    def generate_sort_metric(self, value: float) -> float:
        return value

    def load_checkpoint(self, fs: FSBase) -> None:
        return

    def save_checkpoint(self, fs: FSBase) -> None:
        return

    def initialize(self) -> None:
        return

    def finalize(self) -> None:
        return

    def run_single_iteration(self) -> TrialReport:
        raise NotImplementedError

    def run_single_rung(self, budget: float) -> TrialReport:
        used = 0.0
        while True:
            current_report = self.run_single_iteration()
            used += current_report.cost
            if used >= budget:
                return current_report.with_cost(used)

    def run(self, trial: Trial, judge: TrialJudge, checkpoint_basedir_fs: FSBase) -> None:
        checkpoint = Checkpoint(checkpoint_basedir_fs.makedir(trial.trial_id, recreate=True))
        if not judge.can_accept(trial):
            return
        self._current_trial = trial
        self.initialize()
        try:
            if len(checkpoint) > 0:
                self._rung = int(checkpoint.latest.readtext('__RUNG__')) + 1
                self.load_checkpoint(checkpoint.latest)
            budget = judge.get_budget(trial, self.rung)
            while budget > 0:
                report = self.run_single_rung(budget)
                report = report.with_rung(self.rung).with_sort_metric(self.generate_sort_metric(report.metric))
                decision = judge.judge(report)
                if decision.should_checkpoint:
                    with checkpoint.create() as fs:
                        fs.writetext('__RUNG__', str(self.rung))
                        self.save_checkpoint(fs)
                budget = decision.budget
                self._rung += 1
        finally:
            self.finalize()