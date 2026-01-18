from __future__ import annotations
import re
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import (
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import (
from xarray.core.utils import emit_user_level_warning
def _generate_anchored_deprecated_frequencies(deprecated, recommended):
    pairs = {}
    for abbreviation in _MONTH_ABBREVIATIONS.values():
        anchored_deprecated = f'{deprecated}-{abbreviation}'
        anchored_recommended = f'{recommended}-{abbreviation}'
        pairs[anchored_deprecated] = anchored_recommended
    return pairs