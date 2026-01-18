import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
def _get_cancast_table():
    table = textwrap.dedent('\n        X ? b h i l q B H I L Q e f d g F D G S U V O M m\n        ? # = = = = = = = = = = = = = = = = = = = = = . =\n        b . # = = = = . . . . . = = = = = = = = = = = . =\n        h . ~ # = = = . . . . . ~ = = = = = = = = = = . =\n        i . ~ ~ # = = . . . . . ~ ~ = = ~ = = = = = = . =\n        l . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =\n        q . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =\n        B . ~ = = = = # = = = = = = = = = = = = = = = . =\n        H . ~ ~ = = = ~ # = = = ~ = = = = = = = = = = . =\n        I . ~ ~ ~ = = ~ ~ # = = ~ ~ = = ~ = = = = = = . =\n        L . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~\n        Q . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~\n        e . . . . . . . . . . . # = = = = = = = = = = . .\n        f . . . . . . . . . . . ~ # = = = = = = = = = . .\n        d . . . . . . . . . . . ~ ~ # = ~ = = = = = = . .\n        g . . . . . . . . . . . ~ ~ ~ # ~ ~ = = = = = . .\n        F . . . . . . . . . . . . . . . # = = = = = = . .\n        D . . . . . . . . . . . . . . . ~ # = = = = = . .\n        G . . . . . . . . . . . . . . . ~ ~ # = = = = . .\n        S . . . . . . . . . . . . . . . . . . # = = = . .\n        U . . . . . . . . . . . . . . . . . . . # = = . .\n        V . . . . . . . . . . . . . . . . . . . . # = . .\n        O . . . . . . . . . . . . . . . . . . . . = # . .\n        M . . . . . . . . . . . . . . . . . . . . = = # .\n        m . . . . . . . . . . . . . . . . . . . . = = . #\n        ').strip().split('\n')
    dtypes = [type(np.dtype(c)) for c in table[0][2::2]]
    convert_cast = {'.': Casting.unsafe, '~': Casting.same_kind, '=': Casting.safe, '#': Casting.equiv, ' ': -1}
    cancast = {}
    for from_dt, row in zip(dtypes, table[1:]):
        cancast[from_dt] = {}
        for to_dt, c in zip(dtypes, row[2::2]):
            cancast[from_dt][to_dt] = convert_cast[c]
    return cancast