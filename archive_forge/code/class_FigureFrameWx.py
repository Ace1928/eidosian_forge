import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
class FigureFrameWx(wx.Frame):

    def __init__(self, num, fig, *, canvas_class):
        if wx.Platform == '__WXMSW__':
            pos = wx.DefaultPosition
        else:
            pos = wx.Point(20, 20)
        super().__init__(parent=None, id=-1, pos=pos)
        _log.debug('%s - __init__()', type(self))
        _set_frame_icon(self)
        self.canvas = canvas_class(self, -1, fig)
        manager = FigureManagerWx(self.canvas, num, self)
        toolbar = self.canvas.manager.toolbar
        if toolbar is not None:
            self.SetToolBar(toolbar)
        w, h = map(math.ceil, fig.bbox.size)
        self.canvas.SetInitialSize(wx.Size(w, h))
        self.canvas.SetMinSize((2, 2))
        self.canvas.SetFocus()
        self.Fit()
        self.Bind(wx.EVT_CLOSE, self._on_close)

    def _on_close(self, event):
        _log.debug('%s - on_close()', type(self))
        CloseEvent('close_event', self.canvas)._process()
        self.canvas.stop_event_loop()
        self.canvas.manager.frame = None
        Gcf.destroy(self.canvas.manager)
        try:
            self.canvas.mpl_disconnect(self.canvas.toolbar._id_drag)
        except AttributeError:
            pass
        event.Skip()