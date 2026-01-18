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
def _hook_patch_docstrings(backend):
    from . import _patch_hvplot_docstrings
    _patch_hvplot_docstrings()