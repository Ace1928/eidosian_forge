from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune import Monitor, TrialReport, TrialReportLogger, parse_monitor
class PrintBest(Monitor):

    def __init__(self):
        super().__init__()
        self._lock = SerializableRLock()
        self._bins: Dict[str, '_ReportBin'] = {}

    def on_report(self, report: TrialReport) -> None:
        with self._lock:
            key = str(report.trial.keys)
            if key not in self._bins:
                self._bins[key] = _ReportBin(new_best_only=True)
            rbin = self._bins[key]
        if rbin.on_report(report):
            print(report.trial.keys, report.metric, report)