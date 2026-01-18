from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class RaisingObject:

    def __init__(self, msg='I will raise inside Cython') -> None:
        super().__init__()
        self.msg = msg

    def __eq__(self, other):
        raise RaisingObjectException(self.msg)