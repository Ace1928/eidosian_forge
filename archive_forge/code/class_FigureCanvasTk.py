import uuid
import weakref
from contextlib import contextmanager
import logging
import math
import os.path
import pathlib
import sys
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
from tkinter.simpledialog import SimpleDialog
import numpy as np
from PIL import Image, ImageTk
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook, _c_internal_utils
from matplotlib.backend_bases import (
from matplotlib._pylab_helpers import Gcf
from . import _tkagg
class FigureCanvasTk(FigureCanvasBase):
    required_interactive_framework = 'tk'
    manager_class = _api.classproperty(lambda cls: FigureManagerTk)

    def __init__(self, figure=None, master=None):
        super().__init__(figure)
        self._idle_draw_id = None
        self._event_loop_id = None
        w, h = self.get_width_height(physical=True)
        self._tkcanvas = tk.Canvas(master=master, background='white', width=w, height=h, borderwidth=0, highlightthickness=0)
        self._tkphoto = tk.PhotoImage(master=self._tkcanvas, width=w, height=h)
        self._tkcanvas_image_region = self._tkcanvas.create_image(w // 2, h // 2, image=self._tkphoto)
        self._tkcanvas.bind('<Configure>', self.resize)
        if sys.platform == 'win32':
            self._tkcanvas.bind('<Map>', self._update_device_pixel_ratio)
        self._tkcanvas.bind('<Key>', self.key_press)
        self._tkcanvas.bind('<Motion>', self.motion_notify_event)
        self._tkcanvas.bind('<Enter>', self.enter_notify_event)
        self._tkcanvas.bind('<Leave>', self.leave_notify_event)
        self._tkcanvas.bind('<KeyRelease>', self.key_release)
        for name in ['<Button-1>', '<Button-2>', '<Button-3>']:
            self._tkcanvas.bind(name, self.button_press_event)
        for name in ['<Double-Button-1>', '<Double-Button-2>', '<Double-Button-3>']:
            self._tkcanvas.bind(name, self.button_dblclick_event)
        for name in ['<ButtonRelease-1>', '<ButtonRelease-2>', '<ButtonRelease-3>']:
            self._tkcanvas.bind(name, self.button_release_event)
        for name in ('<Button-4>', '<Button-5>'):
            self._tkcanvas.bind(name, self.scroll_event)
        root = self._tkcanvas.winfo_toplevel()
        weakself = weakref.ref(self)
        weakroot = weakref.ref(root)

        def scroll_event_windows(event):
            self = weakself()
            if self is None:
                root = weakroot()
                if root is not None:
                    root.unbind('<MouseWheel>', scroll_event_windows_id)
                return
            return self.scroll_event_windows(event)
        scroll_event_windows_id = root.bind('<MouseWheel>', scroll_event_windows, '+')

        def filter_destroy(event):
            self = weakself()
            if self is None:
                root = weakroot()
                if root is not None:
                    root.unbind('<Destroy>', filter_destroy_id)
                return
            if event.widget is self._tkcanvas:
                CloseEvent('close_event', self)._process()
        filter_destroy_id = root.bind('<Destroy>', filter_destroy, '+')
        self._tkcanvas.focus_set()
        self._rubberband_rect_black = None
        self._rubberband_rect_white = None

    def _update_device_pixel_ratio(self, event=None):
        ratio = round(self._tkcanvas.tk.call('tk', 'scaling') / (96 / 72), 2)
        if self._set_device_pixel_ratio(ratio):
            w, h = self.get_width_height(physical=True)
            self._tkcanvas.configure(width=w, height=h)

    def resize(self, event):
        width, height = (event.width, event.height)
        dpival = self.figure.dpi
        winch = width / dpival
        hinch = height / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        self._tkcanvas.delete(self._tkcanvas_image_region)
        self._tkphoto.configure(width=int(width), height=int(height))
        self._tkcanvas_image_region = self._tkcanvas.create_image(int(width / 2), int(height / 2), image=self._tkphoto)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    def draw_idle(self):
        if self._idle_draw_id:
            return

        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = None
        self._idle_draw_id = self._tkcanvas.after_idle(idle_draw)

    def get_tk_widget(self):
        """
        Return the Tk widget used to implement FigureCanvasTkAgg.

        Although the initial implementation uses a Tk canvas,  this routine
        is intended to hide that fact.
        """
        return self._tkcanvas

    def _event_mpl_coords(self, event):
        return (self._tkcanvas.canvasx(event.x), self.figure.bbox.height - self._tkcanvas.canvasy(event.y))

    def motion_notify_event(self, event):
        MouseEvent('motion_notify_event', self, *self._event_mpl_coords(event), modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def enter_notify_event(self, event):
        LocationEvent('figure_enter_event', self, *self._event_mpl_coords(event), modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def leave_notify_event(self, event):
        LocationEvent('figure_leave_event', self, *self._event_mpl_coords(event), modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def button_press_event(self, event, dblclick=False):
        self._tkcanvas.focus_set()
        num = getattr(event, 'num', None)
        if sys.platform == 'darwin':
            num = {2: 3, 3: 2}.get(num, num)
        MouseEvent('button_press_event', self, *self._event_mpl_coords(event), num, dblclick=dblclick, modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def button_dblclick_event(self, event):
        self.button_press_event(event, dblclick=True)

    def button_release_event(self, event):
        num = getattr(event, 'num', None)
        if sys.platform == 'darwin':
            num = {2: 3, 3: 2}.get(num, num)
        MouseEvent('button_release_event', self, *self._event_mpl_coords(event), num, modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def scroll_event(self, event):
        num = getattr(event, 'num', None)
        step = 1 if num == 4 else -1 if num == 5 else 0
        MouseEvent('scroll_event', self, *self._event_mpl_coords(event), step=step, modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    def scroll_event_windows(self, event):
        """MouseWheel event processor"""
        w = event.widget.winfo_containing(event.x_root, event.y_root)
        if w != self._tkcanvas:
            return
        x = self._tkcanvas.canvasx(event.x_root - w.winfo_rootx())
        y = self.figure.bbox.height - self._tkcanvas.canvasy(event.y_root - w.winfo_rooty())
        step = event.delta / 120
        MouseEvent('scroll_event', self, x, y, step=step, modifiers=self._mpl_modifiers(event), guiEvent=event)._process()

    @staticmethod
    def _mpl_modifiers(event, *, exclude=None):
        modifiers = [('ctrl', 1 << 2, 'control'), ('alt', 1 << 17, 'alt'), ('shift', 1 << 0, 'shift')] if sys.platform == 'win32' else [('ctrl', 1 << 2, 'control'), ('alt', 1 << 4, 'alt'), ('shift', 1 << 0, 'shift'), ('cmd', 1 << 3, 'cmd')] if sys.platform == 'darwin' else [('ctrl', 1 << 2, 'control'), ('alt', 1 << 3, 'alt'), ('shift', 1 << 0, 'shift'), ('super', 1 << 6, 'super')]
        return [name for name, mask, key in modifiers if event.state & mask and exclude != key]

    def _get_key(self, event):
        unikey = event.char
        key = cbook._unikey_or_keysym_to_mplkey(unikey, event.keysym)
        if key is not None:
            mods = self._mpl_modifiers(event, exclude=key)
            if 'shift' in mods and unikey:
                mods.remove('shift')
            return '+'.join([*mods, key])

    def key_press(self, event):
        KeyEvent('key_press_event', self, self._get_key(event), *self._event_mpl_coords(event), guiEvent=event)._process()

    def key_release(self, event):
        KeyEvent('key_release_event', self, self._get_key(event), *self._event_mpl_coords(event), guiEvent=event)._process()

    def new_timer(self, *args, **kwargs):
        return TimerTk(self._tkcanvas, *args, **kwargs)

    def flush_events(self):
        self._tkcanvas.update()

    def start_event_loop(self, timeout=0):
        if timeout > 0:
            milliseconds = int(1000 * timeout)
            if milliseconds > 0:
                self._event_loop_id = self._tkcanvas.after(milliseconds, self.stop_event_loop)
            else:
                self._event_loop_id = self._tkcanvas.after_idle(self.stop_event_loop)
        self._tkcanvas.mainloop()

    def stop_event_loop(self):
        if self._event_loop_id:
            self._tkcanvas.after_cancel(self._event_loop_id)
            self._event_loop_id = None
        self._tkcanvas.quit()

    def set_cursor(self, cursor):
        try:
            self._tkcanvas.configure(cursor=cursord[cursor])
        except tkinter.TclError:
            pass