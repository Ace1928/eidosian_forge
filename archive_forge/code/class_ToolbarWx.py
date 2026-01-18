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
class ToolbarWx(ToolContainerBase, wx.ToolBar):

    def __init__(self, toolmanager, parent=None, style=wx.TB_BOTTOM):
        if parent is None:
            parent = toolmanager.canvas.GetParent()
        ToolContainerBase.__init__(self, toolmanager)
        wx.ToolBar.__init__(self, parent, -1, style=style)
        self._space = self.AddStretchableSpace()
        self._label_text = wx.StaticText(self, style=wx.ALIGN_RIGHT)
        self.AddControl(self._label_text)
        self._toolitems = {}
        self._groups = {}

    def _get_tool_pos(self, tool):
        """
        Find the position (index) of a wx.ToolBarToolBase in a ToolBar.

        ``ToolBar.GetToolPos`` is not useful because wx assigns the same Id to
        all Separators and StretchableSpaces.
        """
        pos, = [pos for pos in range(self.ToolsCount) if self.GetToolByPos(pos) == tool]
        return pos

    def add_toolitem(self, name, group, position, image_file, description, toggle):
        if group not in self._groups:
            self._groups[group] = self.InsertSeparator(self._get_tool_pos(self._space))
        sep = self._groups[group]
        seps = [t for t in map(self.GetToolByPos, range(self.ToolsCount)) if t.IsSeparator() and (not t.IsStretchableSpace())]
        if position >= 0:
            start = 0 if sep == seps[0] else self._get_tool_pos(seps[seps.index(sep) - 1]) + 1
        else:
            start = self._get_tool_pos(sep) + 1
        idx = start + position
        if image_file:
            bmp = NavigationToolbar2Wx._icon(image_file)
            kind = wx.ITEM_NORMAL if not toggle else wx.ITEM_CHECK
            tool = self.InsertTool(idx, -1, name, bmp, wx.NullBitmap, kind, description or '')
        else:
            size = (self.GetTextExtent(name)[0] + 10, -1)
            if toggle:
                control = wx.ToggleButton(self, -1, name, size=size)
            else:
                control = wx.Button(self, -1, name, size=size)
            tool = self.InsertControl(idx, control, label=name)
        self.Realize()

        def handler(event):
            self.trigger_tool(name)
        if image_file:
            self.Bind(wx.EVT_TOOL, handler, tool)
        else:
            control.Bind(wx.EVT_LEFT_DOWN, handler)
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((tool, handler))

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for tool, handler in self._toolitems[name]:
            if not tool.IsControl():
                self.ToggleTool(tool.Id, toggled)
            else:
                tool.GetControl().SetValue(toggled)
        self.Refresh()

    def remove_toolitem(self, name):
        for tool, handler in self._toolitems[name]:
            self.DeleteTool(tool.Id)
        del self._toolitems[name]

    def set_message(self, s):
        self._label_text.SetLabel(s)