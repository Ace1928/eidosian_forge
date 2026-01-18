import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
def _generate_evaluated_dict(file_name, nrows, ncols):
    result = {}
    keys = [f'col{x}' for x in range(ncols)]
    with open(file_name, mode='w') as _file:
        for i in range(nrows):
            data = np.random.rand(ncols)
            for idx, key in enumerate(keys):
                result[key] = data[idx]
            _file.write(str(result))
            _file.write('\n')