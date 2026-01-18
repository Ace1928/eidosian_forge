from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def _get_horizontal_grid():
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[:, 2])
    return (ax1, ax2)