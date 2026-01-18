import numpy as np
import pytest
from pandas.compat import (
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
from pandas.core.arrays.integer import (
from pandas.tests.extension import base
def _check_divmod_op(self, s, op, other, exc=None):
    super()._check_divmod_op(s, op, other, None)