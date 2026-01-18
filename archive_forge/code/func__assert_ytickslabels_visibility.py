import pytest
from pandas import DataFrame
from pandas.tests.plotting.common import _check_visible
def _assert_ytickslabels_visibility(self, axes, expected):
    for ax, exp in zip(axes, expected):
        _check_visible(ax.get_yticklabels(), visible=exp)