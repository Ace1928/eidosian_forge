import base64
import os
from contextlib import contextmanager
from io import BytesIO
from itertools import chain
from tempfile import NamedTemporaryFile
import matplotlib as mpl
import param
from matplotlib import pyplot as plt
from param.parameterized import bothmethod
from ...core import HoloMap
from ...core.options import Store
from ..renderer import HTML_TAGS, MIME_TYPES, Renderer
from .util import get_old_rcparams, get_tight_bbox
def _compute_bbox(self, fig, kw):
    """
        Compute the tight bounding box for each figure once, reducing
        number of required canvas draw calls from N*2 to N+1 as a
        function of the number of frames.

        Tight bounding box computing code here mirrors:
        matplotlib.backend_bases.FigureCanvasBase.print_figure
        as it hasn't been factored out as a function.
        """
    fig_id = id(fig)
    if kw['bbox_inches'] == 'tight':
        if fig_id not in MPLRenderer.drawn:
            fig.set_dpi(self.dpi)
            fig.canvas.draw()
            extra_artists = kw.pop('bbox_extra_artists', [])
            pad = mpl.rcParams['savefig.pad_inches']
            bbox_inches = get_tight_bbox(fig, extra_artists, pad=pad)
            MPLRenderer.drawn[fig_id] = bbox_inches
            kw['bbox_inches'] = bbox_inches
        else:
            kw['bbox_inches'] = MPLRenderer.drawn[fig_id]
    return kw