from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune import Monitor, TrialReport, TrialReportLogger, parse_monitor
class NotebookSimpleRungs(NotebookSimpleChart):

    def __init__(self, interval: Any='1sec'):
        super().__init__(interval, best_only=False, always_update=True)

    def plot(self, df: pd.DataFrame) -> None:
        import seaborn as sns
        sns.lineplot(data=df, x='rung', y='metric', hue='id', marker='o', legend=False)