from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import logging
from fontTools.designspaceLib import (
@dataclass
class StatNames:
    """Name data generated from the STAT table information."""
    familyNames: Dict[str, str]
    styleNames: Dict[str, str]
    postScriptFontName: Optional[str]
    styleMapFamilyNames: Dict[str, str]
    styleMapStyleName: Optional[RibbiStyle]