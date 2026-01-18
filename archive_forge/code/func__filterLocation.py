from __future__ import annotations
import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast
from fontTools.designspaceLib import (
from fontTools.designspaceLib.statNames import StatNames, getStatNames
from fontTools.designspaceLib.types import (
def _filterLocation(userRegion: Region, location: Dict[str, float]) -> Dict[str, float]:
    return {name: value for name, value in location.items() if name in userRegion and isinstance(userRegion[name], Range)}