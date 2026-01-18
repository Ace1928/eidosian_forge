from __future__ import annotations
from typing import Dict, List, Union
import fontTools.otlLib.builder
from fontTools.designspaceLib import (
from fontTools.designspaceLib.types import Region, getVFUserRegion, locationInRegion
from fontTools.ttLib import TTFont
def _labelToFlags(label: Union[AxisLabelDescriptor, LocationLabelDescriptor]) -> int:
    flags = 0
    if label.olderSibling:
        flags |= 1
    if label.elidable:
        flags |= 2
    return flags