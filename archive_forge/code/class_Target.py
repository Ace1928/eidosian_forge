from dataclasses import dataclass
from typing import List, Optional
@dataclass
class Target:
    """Defines a Grafana target (time-series query) within a panel.

    A panel will have one or more targets. By default, all targets are rendered as
    stacked area charts, with the exception of legend="MAX", which is rendered as
    a blue dotted line. Any legend="FINISHED|FAILED|DEAD|REMOVED" series will also be
    rendered hidden by default.

    Attributes:
        expr: The prometheus query to evaluate.
        legend: The legend string to format for each time-series.
    """
    expr: str
    legend: str