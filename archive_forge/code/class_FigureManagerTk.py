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
class FigureManagerTk(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : tk.Toolbar
        The tk.Toolbar
    window : tk.Window
        The tk.Window
    """
    _owns_mainloop = False

    def __init__(self, canvas, num, window):
        self.window = window
        super().__init__(canvas, num)
        self.window.withdraw()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        window_frame = int(window.wm_frame(), 16)
        self._window_dpi = tk.IntVar(master=window, value=96, name=f'window_dpi{window_frame}')
        self._window_dpi_cbname = ''
        if _tkagg.enable_dpi_awareness(window_frame, window.tk.interpaddr()):
            self._window_dpi_cbname = self._window_dpi.trace_add('write', self._update_window_dpi)
        self._shown = False

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        with _restore_foreground_window_at_end():
            if cbook._get_running_interactive_framework() is None:
                cbook._setup_new_guiapp()
                _c_internal_utils.Win32_SetProcessDpiAwareness_max()
            window = tk.Tk(className='matplotlib')
            window.withdraw()
            icon_fname = str(cbook._get_data_path('images/matplotlib.png'))
            icon_img = ImageTk.PhotoImage(file=icon_fname, master=window)
            icon_fname_large = str(cbook._get_data_path('images/matplotlib_large.png'))
            icon_img_large = ImageTk.PhotoImage(file=icon_fname_large, master=window)
            window.iconphoto(False, icon_img_large, icon_img)
            canvas = canvas_class(figure, master=window)
            manager = cls(canvas, num, window)
            if mpl.is_interactive():
                manager.show()
                canvas.draw_idle()
            return manager

    @classmethod
    def start_main_loop(cls):
        managers = Gcf.get_all_fig_managers()
        if managers:
            first_manager = managers[0]
            manager_class = type(first_manager)
            if manager_class._owns_mainloop:
                return
            manager_class._owns_mainloop = True
            try:
                first_manager.window.mainloop()
            finally:
                manager_class._owns_mainloop = False

    def _update_window_dpi(self, *args):
        newdpi = self._window_dpi.get()
        self.window.call('tk', 'scaling', newdpi / 72)
        if self.toolbar and hasattr(self.toolbar, '_rescale'):
            self.toolbar._rescale()
        self.canvas._update_device_pixel_ratio()

    def resize(self, width, height):
        max_size = 1400000
        if (width > max_size or height > max_size) and sys.platform == 'linux':
            raise ValueError(f'You have requested to resize the Tk window to ({width}, {height}), one of which is bigger than {max_size}.  At larger sizes xorg will either exit with an error on newer versions (~1.20) or cause corruption on older version (~1.19).  We do not expect a window over a million pixel wide or tall to be intended behavior.')
        self.canvas._tkcanvas.configure(width=width, height=height)

    def show(self):
        with _restore_foreground_window_at_end():
            if not self._shown:

                def destroy(*args):
                    Gcf.destroy(self)
                self.window.protocol('WM_DELETE_WINDOW', destroy)
                self.window.deiconify()
                self.canvas._tkcanvas.focus_set()
            else:
                self.canvas.draw_idle()
            if mpl.rcParams['figure.raise_window']:
                self.canvas.manager.window.attributes('-topmost', 1)
                self.canvas.manager.window.attributes('-topmost', 0)
            self._shown = True

    def destroy(self, *args):
        if self.canvas._idle_draw_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._idle_draw_id)
        if self.canvas._event_loop_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._event_loop_id)
        if self._window_dpi_cbname:
            self._window_dpi.trace_remove('write', self._window_dpi_cbname)

        def delayed_destroy():
            self.window.destroy()
            if self._owns_mainloop and (not Gcf.get_num_fig_managers()):
                self.window.quit()
        if cbook._get_running_interactive_framework() == 'tk':
            self.window.after_idle(self.window.after, 0, delayed_destroy)
        else:
            self.window.update()
            delayed_destroy()

    def get_window_title(self):
        return self.window.wm_title()

    def set_window_title(self, title):
        self.window.wm_title(title)

    def full_screen_toggle(self):
        is_fullscreen = bool(self.window.attributes('-fullscreen'))
        self.window.attributes('-fullscreen', not is_fullscreen)