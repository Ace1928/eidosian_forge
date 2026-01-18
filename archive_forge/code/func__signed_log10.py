from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def _signed_log10(x):
    return np.round(np.sign(x) * np.log10(np.sign(x) * x)).astype(int)