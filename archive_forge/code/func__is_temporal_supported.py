from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
def _is_temporal_supported(self, opname, pa_dtype):
    return (opname in ('__add__', '__radd__') or (opname in ('__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__') and (not pa_version_under14p0))) and pa.types.is_duration(pa_dtype) or (opname in ('__sub__', '__rsub__') and pa.types.is_temporal(pa_dtype))