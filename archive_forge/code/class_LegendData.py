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
class LegendData(traitlets.HasTraits):
    description = Unicode('Label').tag(sync=True)
    icon = Unicode('mdi-chart-bubble').tag(sync=True)
    description_color = Unicode().tag(sync=True)

    @traitlets.default('description_color')
    def _description_color(self):
        value = self.color
        while value.ndim >= 1 and (not isinstance(value, str)):
            value = value[0]
        value = value.item()
        if not isinstance(value, str):
            value = 'red'
        return value