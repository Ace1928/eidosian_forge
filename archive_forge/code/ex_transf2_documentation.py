import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (

    >>> plt.plot(edges[:-1], squaretg.pdf(edges[:-1], 10), 'r')
    [<matplotlib.lines.Line2D object at 0x06EBFDB0>]
    >>> plt.fill(edges[4:8], squaretg.pdf(edges[4:8], 10), 'r')
    [<matplotlib.patches.Polygon object at 0x0725BA90>]
    >>> plt.show()
    >>> plt.fill_between(edges[4:8], squaretg.pdf(edges[4:8], 10), y2=0, 'r')
    SyntaxError: non-keyword arg after keyword arg (<console>, line 1)
    >>> plt.fill_between(edges[4:8], squaretg.pdf(edges[4:8], 10), 0, 'r')
    Traceback (most recent call last):
    AttributeError: 'module' object has no attribute 'fill_between'
    >>> fig = figure()
    Traceback (most recent call last):
    NameError: name 'figure' is not defined
    >>> ax1 = fig.add_subplot(311)
    Traceback (most recent call last):
    NameError: name 'fig' is not defined
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(111)
    >>> ax1.fill_between(edges[4:8], squaretg.pdf(edges[4:8], 10), 0, 'r')
    Traceback (most recent call last):
    AttributeError: 'AxesSubplot' object has no attribute 'fill_between'
    >>> ax1.fill(edges[4:8], squaretg.pdf(edges[4:8], 10), 0, 'r')
    Traceback (most recent call last):
    