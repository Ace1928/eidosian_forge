from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import logging
from fontTools.designspaceLib import (
def _getSortedAxisLabels(axes: list[Union[AxisDescriptor, DiscreteAxisDescriptor]]) -> Dict[str, list[AxisLabelDescriptor]]:
    """Returns axis labels sorted by their ordering, with unordered ones appended as
    they are listed."""
    sortedAxes = sorted((axis for axis in axes if axis.axisOrdering is not None), key=lambda a: a.axisOrdering)
    sortedLabels: Dict[str, list[AxisLabelDescriptor]] = {axis.name: axis.axisLabels for axis in sortedAxes}
    for axis in axes:
        if axis.axisOrdering is None:
            sortedLabels[axis.name] = axis.axisLabels
    return sortedLabels