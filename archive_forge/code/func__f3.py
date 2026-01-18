import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@deprecate_kwarg('old', 'new', _f3_mapping)
def _f3(new=0):
    return new