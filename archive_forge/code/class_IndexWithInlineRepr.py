from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
class IndexWithInlineRepr(CustomIndex):

    def _repr_inline_(self, max_width: int):
        return f'CustomIndex[{', '.join(self.names)}]'