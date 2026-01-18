import inspect
import textwrap
import param
import panel as _pn
import holoviews as _hv
from holoviews import Store, render  # noqa
from .converter import HoloViewsConverter
from .interactive import Interactive
from .ui import explorer  # noqa
from .utilities import hvplot_extension, output, save, show # noqa
from .plotting import (hvPlot, hvPlotTabular,  # noqa
class _PatchHvplotDocstrings:

    def __init__(self):
        signatures = {}
        for cls in [hvPlot, hvPlotTabular]:
            for _kind in HoloViewsConverter._kind_mapping:
                if hasattr(cls, _kind):
                    method = getattr(cls, _kind)
                    sig = inspect.signature(method)
                    signatures[cls, _kind] = sig
        self.orig_signatures = signatures

    def __call__(self):
        for cls in [hvPlot, hvPlotTabular]:
            for _kind in HoloViewsConverter._kind_mapping:
                if hasattr(cls, _kind):
                    signature = self.orig_signatures[cls, _kind]
                    _patch_doc(cls, _kind, signature=signature)