from __future__ import absolute_import
import logging
import warnings
import numpy as np
import ipywidgets as widgets  # we should not have widgets under two names
import ipywebrtc
import pythreejs
import traitlets
from traitlets import Unicode, Integer
from traittypes import Array
from bqplot import scales
import ipyvolume
import ipyvolume as ipv  # we should not have ipyvolume under two names either
import ipyvolume._version
from ipyvolume.traittypes import Image
from ipyvolume.serialize import (
from ipyvolume.transferfunction import TransferFunction
from ipyvolume.utils import debounced, grid_slice, reduce_size
def _typefix(value):
    if isinstance(value, (list, tuple)):
        return [_typefix(k) for k in value]
    else:
        try:
            value = value.item()
        except:
            pass
        return value