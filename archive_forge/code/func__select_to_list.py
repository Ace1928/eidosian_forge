import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
@staticmethod
def _select_to_list(dos_collection: Sequence[D], info_selection: Dict[str, str], negative: bool=False) -> List[D]:
    query = set(info_selection.items())
    if negative:
        return [data for data in dos_collection if not query.issubset(set(data.info.items()))]
    else:
        return [data for data in dos_collection if query.issubset(set(data.info.items()))]