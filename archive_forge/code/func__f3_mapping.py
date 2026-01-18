import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
def _f3_mapping(x):
    return x + 1