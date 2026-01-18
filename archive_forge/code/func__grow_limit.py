from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def _grow_limit(limits, values):
    if isinstance(values, (tuple, list)) and len(values) == 2:
        newvmin, newvmax = values
    else:
        try:
            values[0]
        except TypeError:
            newvmin = values
            newvmax = values
        except IndexError:
            newvmin = values
            newvmax = values
        else:
            finites = np.isfinite(values)
            newvmin = np.min(values[finites])
            newvmax = np.max(values[finites])
    if limits is None:
        return (newvmin, newvmax)
    else:
        vmin, vmax = limits
        return (min(newvmin, vmin), max(newvmax, vmax))