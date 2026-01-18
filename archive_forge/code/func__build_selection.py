import numpy as np
from ...core.options import Store
from ...core.overlay import NdOverlay, Overlay
from ...selection import OverlaySelectionDisplay, SelectionDisplay
def _build_selection(self, el, exprs, **kwargs):
    opts = {}
    if exprs[1]:
        mask = exprs[1].apply(el.dataset, expanded=True, flat=True)
        opts['selected'] = list(np.where(mask)[0])
    return el.opts(clone=True, backend='bokeh', **opts)