import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class SimpleDataFrameSubClass(DataFrame):
    """A subclass of DataFrame that does not define a constructor."""