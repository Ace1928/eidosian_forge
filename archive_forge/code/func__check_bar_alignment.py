import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def _check_bar_alignment(self, df, kind='bar', stacked=False, subplots=False, align='center', width=0.5, position=0.5):
    axes = df.plot(kind=kind, stacked=stacked, subplots=subplots, align=align, width=width, position=position, grid=True)
    axes = _flatten_visible(axes)
    for ax in axes:
        if kind == 'bar':
            axis = ax.xaxis
            ax_min, ax_max = ax.get_xlim()
            min_edge = min((p.get_x() for p in ax.patches))
            max_edge = max((p.get_x() + p.get_width() for p in ax.patches))
        elif kind == 'barh':
            axis = ax.yaxis
            ax_min, ax_max = ax.get_ylim()
            min_edge = min((p.get_y() for p in ax.patches))
            max_edge = max((p.get_y() + p.get_height() for p in ax.patches))
        else:
            raise ValueError
        tm.assert_almost_equal(ax_min, min_edge - 0.25)
        tm.assert_almost_equal(ax_max, max_edge + 0.25)
        p = ax.patches[0]
        if kind == 'bar' and (stacked is True or subplots is True):
            edge = p.get_x()
            center = edge + p.get_width() * position
        elif kind == 'bar' and stacked is False:
            center = p.get_x() + p.get_width() * len(df.columns) * position
            edge = p.get_x()
        elif kind == 'barh' and (stacked is True or subplots is True):
            center = p.get_y() + p.get_height() * position
            edge = p.get_y()
        elif kind == 'barh' and stacked is False:
            center = p.get_y() + p.get_height() * len(df.columns) * position
            edge = p.get_y()
        else:
            raise ValueError
        assert (axis.get_ticklocs() == np.arange(len(df))).all()
        if align == 'center':
            tm.assert_almost_equal(axis.get_ticklocs()[0], center)
        elif align == 'edge':
            tm.assert_almost_equal(axis.get_ticklocs()[0], edge)
        else:
            raise ValueError
    return axes