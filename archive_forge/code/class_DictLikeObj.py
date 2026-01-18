import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class DictLikeObj:

    def keys(self):
        return ('a',)

    def __getitem__(self, item):
        if item == 'a':
            return dict_val