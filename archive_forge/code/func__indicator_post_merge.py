from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
@final
def _indicator_post_merge(self, result: DataFrame) -> DataFrame:
    result['_left_indicator'] = result['_left_indicator'].fillna(0)
    result['_right_indicator'] = result['_right_indicator'].fillna(0)
    result[self._indicator_name] = Categorical(result['_left_indicator'] + result['_right_indicator'], categories=[1, 2, 3])
    result[self._indicator_name] = result[self._indicator_name].cat.rename_categories(['left_only', 'right_only', 'both'])
    result = result.drop(labels=['_left_indicator', '_right_indicator'], axis=1)
    return result