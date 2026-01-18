from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune import Monitor, TrialReport, TrialReportLogger, parse_monitor
@parse_monitor.candidate(lambda obj: isinstance(obj, str) and obj == 'ts', priority=0.0)
def _ts(obj: str) -> Monitor:
    return NotebookSimpleTimeSeries()