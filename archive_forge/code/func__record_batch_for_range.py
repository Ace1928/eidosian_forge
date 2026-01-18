import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def _record_batch_for_range(schema, n):
    return _record_batch_from_iters(schema, range(n, n + 10), range(n + 1, n + 11))