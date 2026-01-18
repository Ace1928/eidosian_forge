from collections.abc import Mapping
import functools
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, colors, cbook, scale
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed

        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        