from dataclasses import dataclass
from typing import List, Optional
@dataclass
class Panel:
    """Defines a Grafana panel (graph) for the Ray dashboard page.

    A panel contains one or more targets (time-series queries).

    Attributes:
        title: Short name of the graph. Note: please keep this in sync with the title
            definitions in Metrics.tsx.
        description: Long form description of the graph.
        id: Integer id used to reference the graph from Metrics.tsx.
        unit: The unit to display on the y-axis of the graph.
        targets: List of query targets.
        fill: Whether or not the graph will be filled by a color.
        stack: Whether or not the lines in the graph will be stacked.
    """
    title: str
    description: str
    id: int
    unit: str
    targets: List[Target]
    fill: int = 10
    stack: bool = True
    linewidth: int = 1
    grid_pos: Optional[GridPos] = None