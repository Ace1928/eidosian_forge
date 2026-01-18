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
class _FigureCanvasWxBase(FigureCanvasBase, wx.Panel):
    """
    The FigureCanvas contains the figure and does event handling.

    In the wxPython backend, it is derived from wxPanel, and (usually) lives
    inside a frame instantiated by a FigureManagerWx. The parent window
    probably implements a wx.Sizer to control the displayed control size - but
    we give a hint as to our preferred minimum size.
    """
    required_interactive_framework = 'wx'
    _timer_cls = TimerWx
    manager_class = _api.classproperty(lambda cls: FigureManagerWx)
    keyvald = {wx.WXK_CONTROL: 'control', wx.WXK_SHIFT: 'shift', wx.WXK_ALT: 'alt', wx.WXK_CAPITAL: 'caps_lock', wx.WXK_LEFT: 'left', wx.WXK_UP: 'up', wx.WXK_RIGHT: 'right', wx.WXK_DOWN: 'down', wx.WXK_ESCAPE: 'escape', wx.WXK_F1: 'f1', wx.WXK_F2: 'f2', wx.WXK_F3: 'f3', wx.WXK_F4: 'f4', wx.WXK_F5: 'f5', wx.WXK_F6: 'f6', wx.WXK_F7: 'f7', wx.WXK_F8: 'f8', wx.WXK_F9: 'f9', wx.WXK_F10: 'f10', wx.WXK_F11: 'f11', wx.WXK_F12: 'f12', wx.WXK_SCROLL: 'scroll_lock', wx.WXK_PAUSE: 'break', wx.WXK_BACK: 'backspace', wx.WXK_RETURN: 'enter', wx.WXK_INSERT: 'insert', wx.WXK_DELETE: 'delete', wx.WXK_HOME: 'home', wx.WXK_END: 'end', wx.WXK_PAGEUP: 'pageup', wx.WXK_PAGEDOWN: 'pagedown', wx.WXK_NUMPAD0: '0', wx.WXK_NUMPAD1: '1', wx.WXK_NUMPAD2: '2', wx.WXK_NUMPAD3: '3', wx.WXK_NUMPAD4: '4', wx.WXK_NUMPAD5: '5', wx.WXK_NUMPAD6: '6', wx.WXK_NUMPAD7: '7', wx.WXK_NUMPAD8: '8', wx.WXK_NUMPAD9: '9', wx.WXK_NUMPAD_ADD: '+', wx.WXK_NUMPAD_SUBTRACT: '-', wx.WXK_NUMPAD_MULTIPLY: '*', wx.WXK_NUMPAD_DIVIDE: '/', wx.WXK_NUMPAD_DECIMAL: 'dec', wx.WXK_NUMPAD_ENTER: 'enter', wx.WXK_NUMPAD_UP: 'up', wx.WXK_NUMPAD_RIGHT: 'right', wx.WXK_NUMPAD_DOWN: 'down', wx.WXK_NUMPAD_LEFT: 'left', wx.WXK_NUMPAD_PAGEUP: 'pageup', wx.WXK_NUMPAD_PAGEDOWN: 'pagedown', wx.WXK_NUMPAD_HOME: 'home', wx.WXK_NUMPAD_END: 'end', wx.WXK_NUMPAD_INSERT: 'insert', wx.WXK_NUMPAD_DELETE: 'delete'}

    def __init__(self, parent, id, figure=None):
        """
        Initialize a FigureWx instance.

        - Initialize the FigureCanvasBase and wxPanel parents.
        - Set event handlers for resize, paint, and keyboard and mouse
          interaction.
        """
        FigureCanvasBase.__init__(self, figure)
        w, h = map(math.ceil, self.figure.bbox.size)
        wx.Panel.__init__(self, parent, id, size=wx.Size(w, h))
        self.bitmap = wx.Bitmap(w, h)
        _log.debug('%s - __init__() - bitmap w:%d h:%d', type(self), w, h)
        self._isDrawn = False
        self._rubberband_rect = None
        self._rubberband_pen_black = wx.Pen('BLACK', 1, wx.PENSTYLE_SHORT_DASH)
        self._rubberband_pen_white = wx.Pen('WHITE', 1, wx.PENSTYLE_SOLID)
        self.Bind(wx.EVT_SIZE, self._on_size)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_key_down)
        self.Bind(wx.EVT_KEY_UP, self._on_key_up)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_LEFT_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MIDDLE_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_MIDDLE_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_MIDDLE_UP, self._on_mouse_button)
        self.Bind(wx.EVT_RIGHT_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_RIGHT_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_RIGHT_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX1_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX1_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX2_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX2_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX1_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX2_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mouse_wheel)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_ENTER_WINDOW, self._on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)
        self.Bind(wx.EVT_MOUSE_CAPTURE_CHANGED, self._on_capture_lost)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_capture_lost)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(wx.WHITE)

    def Copy_to_Clipboard(self, event=None):
        """Copy bitmap of canvas to system clipboard."""
        bmp_obj = wx.BitmapDataObject()
        bmp_obj.SetBitmap(self.bitmap)
        if not wx.TheClipboard.IsOpened():
            open_success = wx.TheClipboard.Open()
            if open_success:
                wx.TheClipboard.SetData(bmp_obj)
                wx.TheClipboard.Flush()
                wx.TheClipboard.Close()

    def draw_idle(self):
        _log.debug('%s - draw_idle()', type(self))
        self._isDrawn = False
        self.Refresh(eraseBackground=False)

    def flush_events(self):
        wx.Yield()

    def start_event_loop(self, timeout=0):
        if hasattr(self, '_event_loop'):
            raise RuntimeError('Event loop already running')
        timer = wx.Timer(self, id=wx.ID_ANY)
        if timeout > 0:
            timer.Start(int(timeout * 1000), oneShot=True)
            self.Bind(wx.EVT_TIMER, self.stop_event_loop, id=timer.GetId())
        self._event_loop = wx.GUIEventLoop()
        self._event_loop.Run()
        timer.Stop()

    def stop_event_loop(self, event=None):
        if hasattr(self, '_event_loop'):
            if self._event_loop.IsRunning():
                self._event_loop.Exit()
            del self._event_loop

    def _get_imagesave_wildcards(self):
        """Return the wildcard string for the filesave dialog."""
        default_filetype = self.get_default_filetype()
        filetypes = self.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        wildcards = []
        extensions = []
        filter_index = 0
        for i, (name, exts) in enumerate(sorted_filetypes):
            ext_list = ';'.join(['*.%s' % ext for ext in exts])
            extensions.append(exts[0])
            wildcard = f'{name} ({ext_list})|{ext_list}'
            if default_filetype in exts:
                filter_index = i
            wildcards.append(wildcard)
        wildcards = '|'.join(wildcards)
        return (wildcards, extensions, filter_index)

    def gui_repaint(self, drawDC=None):
        """
        Update the displayed image on the GUI canvas, using the supplied
        wx.PaintDC device context.
        """
        _log.debug('%s - gui_repaint()', type(self))
        if not (self and self.IsShownOnScreen()):
            return
        if not drawDC:
            drawDC = wx.ClientDC(self)
        bmp = self.bitmap.ConvertToImage().ConvertToBitmap() if wx.Platform == '__WXMSW__' and isinstance(self.figure.canvas.get_renderer(), RendererWx) else self.bitmap
        drawDC.DrawBitmap(bmp, 0, 0)
        if self._rubberband_rect is not None:
            x0, y0, x1, y1 = map(round, self._rubberband_rect)
            rect = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x0, y0, x0, y1), (x0, y1, x1, y1)]
            drawDC.DrawLineList(rect, self._rubberband_pen_white)
            drawDC.DrawLineList(rect, self._rubberband_pen_black)
    filetypes = {**FigureCanvasBase.filetypes, 'bmp': 'Windows bitmap', 'jpeg': 'JPEG', 'jpg': 'JPEG', 'pcx': 'PCX', 'png': 'Portable Network Graphics', 'tif': 'Tagged Image Format File', 'tiff': 'Tagged Image Format File', 'xpm': 'X pixmap'}

    def _on_paint(self, event):
        """Called when wxPaintEvt is generated."""
        _log.debug('%s - _on_paint()', type(self))
        drawDC = wx.PaintDC(self)
        if not self._isDrawn:
            self.draw(drawDC=drawDC)
        else:
            self.gui_repaint(drawDC=drawDC)
        drawDC.Destroy()

    def _on_size(self, event):
        """
        Called when wxEventSize is generated.

        In this application we attempt to resize to fit the window, so it
        is better to take the performance hit and redraw the whole window.
        """
        _log.debug('%s - _on_size()', type(self))
        sz = self.GetParent().GetSizer()
        if sz:
            si = sz.GetItem(self)
        if sz and si and (not si.Proportion) and (not si.Flag & wx.EXPAND):
            size = self.GetMinSize()
        else:
            size = self.GetClientSize()
            size.IncTo(self.GetMinSize())
        if getattr(self, '_width', None):
            if size == (self._width, self._height):
                return
        self._width, self._height = size
        self._isDrawn = False
        if self._width <= 1 or self._height <= 1:
            return
        self.bitmap = wx.Bitmap(self._width, self._height)
        dpival = self.figure.dpi
        winch = self._width / dpival
        hinch = self._height / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        self.Refresh(eraseBackground=False)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    @staticmethod
    def _mpl_modifiers(event=None, *, exclude=None):
        mod_table = [('ctrl', wx.MOD_CONTROL, wx.WXK_CONTROL), ('alt', wx.MOD_ALT, wx.WXK_ALT), ('shift', wx.MOD_SHIFT, wx.WXK_SHIFT)]
        if event is not None:
            modifiers = event.GetModifiers()
            return [name for name, mod, key in mod_table if modifiers & mod and exclude != key]
        else:
            return [name for name, mod, key in mod_table if wx.GetKeyState(key)]

    def _get_key(self, event):
        keyval = event.KeyCode
        if keyval in self.keyvald:
            key = self.keyvald[keyval]
        elif keyval < 256:
            key = chr(keyval)
            if not event.ShiftDown():
                key = key.lower()
        else:
            return None
        mods = self._mpl_modifiers(event, exclude=keyval)
        if 'shift' in mods and key.isupper():
            mods.remove('shift')
        return '+'.join([*mods, key])

    def _mpl_coords(self, pos=None):
        """
        Convert a wx position, defaulting to the current cursor position, to
        Matplotlib coordinates.
        """
        if pos is None:
            pos = wx.GetMouseState()
            x, y = self.ScreenToClient(pos.X, pos.Y)
        else:
            x, y = (pos.X, pos.Y)
        return (x, self.figure.bbox.height - y)

    def _on_key_down(self, event):
        """Capture key press."""
        KeyEvent('key_press_event', self, self._get_key(event), *self._mpl_coords(), guiEvent=event)._process()
        if self:
            event.Skip()

    def _on_key_up(self, event):
        """Release key."""
        KeyEvent('key_release_event', self, self._get_key(event), *self._mpl_coords(), guiEvent=event)._process()
        if self:
            event.Skip()

    def set_cursor(self, cursor):
        cursor = wx.Cursor(_api.check_getitem({cursors.MOVE: wx.CURSOR_HAND, cursors.HAND: wx.CURSOR_HAND, cursors.POINTER: wx.CURSOR_ARROW, cursors.SELECT_REGION: wx.CURSOR_CROSS, cursors.WAIT: wx.CURSOR_WAIT, cursors.RESIZE_HORIZONTAL: wx.CURSOR_SIZEWE, cursors.RESIZE_VERTICAL: wx.CURSOR_SIZENS}, cursor=cursor))
        self.SetCursor(cursor)
        self.Refresh()

    def _set_capture(self, capture=True):
        """Control wx mouse capture."""
        if self.HasCapture():
            self.ReleaseMouse()
        if capture:
            self.CaptureMouse()

    def _on_capture_lost(self, event):
        """Capture changed or lost"""
        self._set_capture(False)

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

    def _on_mouse_wheel(self, event):
        """Translate mouse wheel events into matplotlib events"""
        x, y = self._mpl_coords(event)
        step = event.LinesPerAction * event.WheelRotation / event.WheelDelta
        event.Skip()
        if wx.Platform == '__WXMAC__':
            if not hasattr(self, '_skipwheelevent'):
                self._skipwheelevent = True
            elif self._skipwheelevent:
                self._skipwheelevent = False
                return
            else:
                self._skipwheelevent = True
        MouseEvent('scroll_event', self, x, y, step=step, modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def _on_motion(self, event):
        """Start measuring on an axis."""
        event.Skip()
        MouseEvent('motion_notify_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def _on_enter(self, event):
        """Mouse has entered the window."""
        event.Skip()
        LocationEvent('figure_enter_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(), guiEvent=event)._process()

    def _on_leave(self, event):
        """Mouse has left the window."""
        event.Skip()
        LocationEvent('figure_leave_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(), guiEvent=event)._process()