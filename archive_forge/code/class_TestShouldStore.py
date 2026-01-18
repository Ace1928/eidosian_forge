from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
class TestShouldStore:

    def test_should_store_categorical(self):
        cat = Categorical(['A', 'B', 'C'])
        df = DataFrame(cat)
        blk = df._mgr.blocks[0]
        assert blk.should_store(cat)
        assert blk.should_store(cat[:-1])
        assert not blk.should_store(cat.as_ordered())
        assert not blk.should_store(np.asarray(cat))