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
def _on_mouse_button(self, event):
    """Start measuring on an axis."""
    event.Skip()
    self._set_capture(event.ButtonDown() or event.ButtonDClick())
    x, y = self._mpl_coords(event)
    button_map = {wx.MOUSE_BTN_LEFT: MouseButton.LEFT, wx.MOUSE_BTN_MIDDLE: MouseButton.MIDDLE, wx.MOUSE_BTN_RIGHT: MouseButton.RIGHT, wx.MOUSE_BTN_AUX1: MouseButton.BACK, wx.MOUSE_BTN_AUX2: MouseButton.FORWARD}
    button = event.GetButton()
    button = button_map.get(button, button)
    modifiers = self._mpl_modifiers(event)
    if event.ButtonDown():
        MouseEvent('button_press_event', self, x, y, button, modifiers=modifiers, guiEvent=event)._process()
    elif event.ButtonDClick():
        MouseEvent('button_press_event', self, x, y, button, dblclick=True, modifiers=modifiers, guiEvent=event)._process()
    elif event.ButtonUp():
        MouseEvent('button_release_event', self, x, y, button, modifiers=modifiers, guiEvent=event)._process()