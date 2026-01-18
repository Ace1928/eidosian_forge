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
class _HelpDialog(wx.Dialog):
    _instance = None
    headers = [('Action', 'Shortcuts', 'Description')]
    widths = [100, 140, 300]

    def __init__(self, parent, help_entries):
        super().__init__(parent, title='Help', style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer = wx.FlexGridSizer(0, 3, 8, 6)
        bold = self.GetFont().MakeBold()
        for r, row in enumerate(self.headers + help_entries):
            for col, width in zip(row, self.widths):
                label = wx.StaticText(self, label=col)
                if r == 0:
                    label.SetFont(bold)
                label.Wrap(width)
                grid_sizer.Add(label, 0, 0, 0)
        sizer.Add(grid_sizer, 0, wx.ALL, 6)
        ok = wx.Button(self, wx.ID_OK)
        sizer.Add(ok, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 8)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()
        self.Bind(wx.EVT_CLOSE, self._on_close)
        ok.Bind(wx.EVT_BUTTON, self._on_close)

    def _on_close(self, event):
        _HelpDialog._instance = None
        self.DestroyLater()
        event.Skip()

    @classmethod
    def show(cls, parent, help_entries):
        if cls._instance:
            cls._instance.Raise()
            return
        cls._instance = cls(parent, help_entries)
        cls._instance.Show()