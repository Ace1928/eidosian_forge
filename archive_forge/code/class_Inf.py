from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
class Inf:

    def __lt__(self, o):
        return False

    def __le__(self, o):
        return isinstance(o, Inf)

    def __gt__(self, o):
        return not isinstance(o, Inf)

    def __ge__(self, o):
        return True

    def __eq__(self, other) -> bool:
        return isinstance(other, Inf)