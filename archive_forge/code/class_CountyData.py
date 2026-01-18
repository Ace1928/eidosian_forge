from __future__ import annotations
import logging # isort:skip
import csv
import xml.etree.ElementTree as et
from math import nan
from typing import TYPE_CHECKING, TypedDict
import numpy as np
from ..util.sampledata import external_path, open_csv
class CountyData(TypedDict):
    name: str
    detailed_name: str
    state: str
    lats: NDArray[np.float64]
    lons: NDArray[np.float64]