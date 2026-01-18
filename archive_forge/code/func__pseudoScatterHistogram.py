from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def _pseudoScatterHistogram(data, spacing=None, shuffle=True, bidir=False):
    """Works by binning points into a histogram and spreading them out to fill the bin.
    
    Faster method, but can produce blocky results.
    """
    inds = np.arange(len(data))
    if shuffle:
        np.random.shuffle(inds)
    data = data[inds]
    if spacing is None:
        spacing = 2.0 * np.std(data) / len(data) ** 0.5
    yvals = np.empty(len(data))
    dmin = data.min()
    dmax = data.max()
    nbins = int((dmax - dmin) / spacing) + 1
    bins = np.linspace(dmin, dmax, nbins)
    dx = bins[1] - bins[0]
    dbins = ((data - bins[0]) / dx).astype(int)
    binCounts = {}
    for i, j in enumerate(dbins):
        c = binCounts.get(j, -1) + 1
        binCounts[j] = c
        yvals[i] = c
    if bidir is True:
        for i in range(nbins):
            yvals[dbins == i] -= binCounts.get(i, 0) * 0.5
    return yvals[np.argsort(inds)]