from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
class _PerTrial:

    def __init__(self, parent: '_PerPartition') -> None:
        self._history: List[TrialReport] = []
        self._parent = parent
        self._active = True

    def can_promote(self, report: TrialReport) -> Tuple[bool, str]:
        reasons: List[str] = []
        if self._active:
            can_accept = self._parent.can_accept(report.trial)
            early_stop = self._parent._parent._trial_early_stop(report, self._history, self._parent._rungs)
            self._active = can_accept and (not early_stop)
            if not can_accept:
                reasons.append("can't accept new")
            if early_stop:
                reasons.append('trial early stop')
        if self._active:
            self._history.append(report)
            can_push = self._parent._rungs[report.rung].push(report)
            if not can_push:
                reasons.append('not best')
            return (can_push, ', '.join(reasons))
        return (False, ', '.join(reasons))

    def judge(self, report: TrialReport) -> TrialDecision:
        if report.rung >= len(self._parent._parent.schedule) - 1:
            self._history.append(report)
            self._parent._rungs[report.rung].push(report)
            return TrialDecision(report, budget=0, should_checkpoint=True, reason='last')
        promote, reason = self.can_promote(report)
        if not promote:
            return TrialDecision(report, budget=0, should_checkpoint=True, reason=reason)
        next_budget = self._parent.get_budget(report.trial, report.rung + 1)
        return TrialDecision(report, budget=next_budget, should_checkpoint=next_budget <= 0 or self._parent._parent.always_checkpoint, reason='' if next_budget > 0 else 'budget==0')