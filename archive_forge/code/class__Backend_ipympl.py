import sys
import types
from warnings import warn
import io
import json
from base64 import b64encode
import matplotlib
import numpy as np
from IPython import get_ipython
from IPython import version_info as ipython_version_info
from IPython.display import HTML, display
from ipython_genutils.py3compat import string_types
from ipywidgets import DOMWidget, widget_serialization
from matplotlib import is_interactive, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import NavigationToolbar2, _Backend, cursors
from matplotlib.backends.backend_webagg_core import (
from PIL import Image
from traitlets import (
from ._version import js_semver
@_Backend.export
class _Backend_ipympl(_Backend):
    FigureCanvas = Canvas
    FigureManager = FigureManager
    _to_show = []
    _draw_called = False

    @staticmethod
    def new_figure_manager_given_figure(num, figure):
        canvas = Canvas(figure)
        if 'nbagg.transparent' in rcParams and rcParams['nbagg.transparent']:
            figure.patch.set_alpha(0)
        manager = FigureManager(canvas, num)
        if is_interactive():
            _Backend_ipympl._to_show.append(figure)
            figure.canvas.draw_idle()

        def destroy(event):
            canvas.mpl_disconnect(cid)
            Gcf.destroy(manager)
        cid = canvas.mpl_connect('close_event', destroy)
        if is_interactive():
            try:
                _Backend_ipympl._to_show.remove(figure)
            except ValueError:
                pass
            _Backend_ipympl._to_show.append(figure)
            _Backend_ipympl._draw_called = True
        return manager

    @staticmethod
    def show(block=None):
        interactive = is_interactive()
        manager = Gcf.get_active()
        if manager is None:
            return
        try:
            display(manager.canvas)
            if hasattr(manager, '_cidgcf'):
                manager.canvas.mpl_disconnect(manager._cidgcf)
            if not interactive:
                Gcf.figs.pop(manager.num, None)
        finally:
            if manager.canvas.figure in _Backend_ipympl._to_show:
                _Backend_ipympl._to_show.remove(manager.canvas.figure)