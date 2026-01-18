from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _check_grid_settings(obj, kinds, kws={}):
    import matplotlib as mpl

    def is_grid_on():
        xticks = mpl.pyplot.gca().xaxis.get_major_ticks()
        yticks = mpl.pyplot.gca().yaxis.get_major_ticks()
        xoff = all((not g.gridline.get_visible() for g in xticks))
        yoff = all((not g.gridline.get_visible() for g in yticks))
        return not (xoff and yoff)
    spndx = 1
    for kind in kinds:
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc('axes', grid=False)
        obj.plot(kind=kind, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc('axes', grid=True)
        obj.plot(kind=kind, grid=False, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()
        if kind not in ['pie', 'hexbin', 'scatter']:
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc('axes', grid=True)
            obj.plot(kind=kind, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc('axes', grid=False)
            obj.plot(kind=kind, grid=True, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()