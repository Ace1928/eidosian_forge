import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
@staticmethod
def _check_weights_and_info(weights: Sequence[Sequence[float]], info: Union[Sequence[Info], None]) -> Sequence[Info]:
    if info is None:
        info = [{} for _ in range(len(weights))]
    elif len(info) != len(weights):
        raise ValueError('Length of info must match number of rows in weights')
    return info