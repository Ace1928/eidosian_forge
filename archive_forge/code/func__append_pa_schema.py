from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import (
from triad.utils.schema import (
def _append_pa_schema(self, other: pa.Schema) -> 'Schema':
    for k, v in zip(other.names, other.types):
        self[k] = v
    return self