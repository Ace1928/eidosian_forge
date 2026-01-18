from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
from matplotlib.dates import (
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
def _get_locators(self, locator, upto):
    if locator is not None:
        major_locator = locator
    elif upto is not None:
        major_locator = AutoDateLocator(minticks=2, maxticks=upto)
    else:
        major_locator = AutoDateLocator(minticks=2, maxticks=6)
    minor_locator = None
    return (major_locator, minor_locator)