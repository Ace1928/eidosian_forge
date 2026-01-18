from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def _impl_locale_comma():
    try:
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
    except locale.Error:
        print('SKIP: Locale de_DE.UTF-8 is not supported on this machine')
        return
    ticks = mticker.ScalarFormatter(useMathText=True, useLocale=True)
    fmt = '$\\mathdefault{%1.1f}$'
    x = ticks._format_maybe_minus_and_locale(fmt, 0.5)
    assert x == '$\\mathdefault{0{,}5}$'
    fmt = ',$\\mathdefault{,%1.1f},$'
    x = ticks._format_maybe_minus_and_locale(fmt, 0.5)
    assert x == ',$\\mathdefault{,0{,}5},$'
    ticks = mticker.ScalarFormatter(useMathText=False, useLocale=True)
    fmt = '%1.1f'
    x = ticks._format_maybe_minus_and_locale(fmt, 0.5)
    assert x == '0,5'