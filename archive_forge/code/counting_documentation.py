from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from typing import TYPE_CHECKING
Given data, return numpy.histogram parameters to define bins.