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
def _generate_4_axes_via_gridspec():
    import matplotlib.pyplot as plt
    gs = mpl.gridspec.GridSpec(2, 2)
    ax_tl = plt.subplot(gs[0, 0])
    ax_ll = plt.subplot(gs[1, 0])
    ax_tr = plt.subplot(gs[0, 1])
    ax_lr = plt.subplot(gs[1, 1])
    return (gs, [ax_tl, ax_ll, ax_tr, ax_lr])