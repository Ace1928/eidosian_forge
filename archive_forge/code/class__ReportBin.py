from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune import Monitor, TrialReport, TrialReportLogger, parse_monitor
class _ReportBin(TrialReportLogger):

    def __init__(self, new_best_only: bool=False):
        super().__init__(new_best_only=new_best_only)
        self._values: List[List[Any]] = []

    def log(self, report: TrialReport) -> None:
        self._values.append([str(report.trial.keys), report.rung, datetime.now(), report.trial_id, report.metric, self.best.metric])

    @property
    def records(self) -> List[List[Any]]:
        with self._lock:
            return list(self._values)