import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
def add_brackets(xloc, yloc, xshift=0, color='r', label=None, options=None):
    """Add two brackets on the memory line plot.

    This function uses the current figure.

    Parameters
    ==========
    xloc: tuple with 2 values
        brackets location (on horizontal axis).
    yloc: tuple with 2 values
        brackets location (on vertical axis)
    xshift: float
        value to subtract to xloc.
    """
    try:
        import pylab as pl
    except ImportError as e:
        print('matplotlib is needed for plotting.')
        print(e)
        sys.exit(1)
    height_ratio = 20.0
    vsize = (pl.ylim()[1] - pl.ylim()[0]) / height_ratio
    hsize = (pl.xlim()[1] - pl.xlim()[0]) / (3.0 * height_ratio)
    bracket_x = pl.asarray([hsize, 0, 0, hsize])
    bracket_y = pl.asarray([vsize, vsize, -vsize, -vsize])
    if label[0] == '_':
        label = ' ' + label
    if options.xlim is None or options.xlim[0] <= xloc[0] - xshift <= options.xlim[1]:
        pl.plot(bracket_x + xloc[0] - xshift, bracket_y + yloc[0], '-' + color, linewidth=2, label=label)
    if options.xlim is None or options.xlim[0] <= xloc[1] - xshift <= options.xlim[1]:
        pl.plot(-bracket_x + xloc[1] - xshift, bracket_y + yloc[1], '-' + color, linewidth=2)