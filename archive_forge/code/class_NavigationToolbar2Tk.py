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
class NavigationToolbar2Tk(NavigationToolbar2, tk.Frame):

    def __init__(self, canvas, window=None, *, pack_toolbar=True):
        """
        Parameters
        ----------
        canvas : `FigureCanvas`
            The figure canvas on which to operate.
        window : tk.Window
            The tk.Window which owns this toolbar.
        pack_toolbar : bool, default: True
            If True, add the toolbar to the parent's pack manager's packing
            list during initialization with ``side="bottom"`` and ``fill="x"``.
            If you want to use the toolbar with a different layout manager, use
            ``pack_toolbar=False``.
        """
        if window is None:
            window = canvas.get_tk_widget().master
        tk.Frame.__init__(self, master=window, borderwidth=2, width=int(canvas.figure.bbox.width), height=50)
        self._buttons = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self._Spacer()
            else:
                self._buttons[text] = button = self._Button(text, str(cbook._get_data_path(f'images/{image_file}.png')), toggle=callback in ['zoom', 'pan'], command=getattr(self, callback))
                if tooltip_text is not None:
                    ToolTip.createToolTip(button, tooltip_text)
        self._label_font = tkinter.font.Font(root=window, size=10)
        label = tk.Label(master=self, font=self._label_font, text='\xa0\n\xa0')
        label.pack(side=tk.RIGHT)
        self.message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, font=self._label_font, textvariable=self.message, justify=tk.RIGHT)
        self._message_label.pack(side=tk.RIGHT)
        NavigationToolbar2.__init__(self, canvas)
        if pack_toolbar:
            self.pack(side=tk.BOTTOM, fill=tk.X)

    def _rescale(self):
        """
        Scale all children of the toolbar to current DPI setting.

        Before this is called, the Tk scaling setting will have been updated to
        match the new DPI. Tk widgets do not update for changes to scaling, but
        all measurements made after the change will match the new scaling. Thus
        this function re-applies all the same sizes in points, which Tk will
        scale correctly to pixels.
        """
        for widget in self.winfo_children():
            if isinstance(widget, (tk.Button, tk.Checkbutton)):
                if hasattr(widget, '_image_file'):
                    NavigationToolbar2Tk._set_image_for_button(self, widget)
                else:
                    pass
            elif isinstance(widget, tk.Frame):
                widget.configure(height='18p')
                widget.pack_configure(padx='3p')
            elif isinstance(widget, tk.Label):
                pass
            else:
                _log.warning('Unknown child class %s', widget.winfo_class)
        self._label_font.configure(size=10)

    def _update_buttons_checked(self):
        for text, mode in [('Zoom', _Mode.ZOOM), ('Pan', _Mode.PAN)]:
            if text in self._buttons:
                if self.mode == mode:
                    self._buttons[text].select()
                else:
                    self._buttons[text].deselect()

    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    def set_message(self, s):
        self.message.set(s)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        if self.canvas._rubberband_rect_white:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_white)
        if self.canvas._rubberband_rect_black:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_black)
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        self.canvas._rubberband_rect_black = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1)
        self.canvas._rubberband_rect_white = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1, outline='white', dash=(3, 3))

    def remove_rubberband(self):
        if self.canvas._rubberband_rect_white:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_white)
            self.canvas._rubberband_rect_white = None
        if self.canvas._rubberband_rect_black:
            self.canvas._tkcanvas.delete(self.canvas._rubberband_rect_black)
            self.canvas._rubberband_rect_black = None

    def _set_image_for_button(self, button):
        """
        Set the image for a button based on its pixel size.

        The pixel size is determined by the DPI scaling of the window.
        """
        if button._image_file is None:
            return
        path_regular = cbook._get_data_path('images', button._image_file)
        path_large = path_regular.with_name(path_regular.name.replace('.png', '_large.png'))
        size = button.winfo_pixels('18p')

        def _get_color(color_name):
            return button.winfo_rgb(button.cget(color_name))

        def _is_dark(color):
            if isinstance(color, str):
                color = _get_color(color)
            return max(color) < 65535 / 2

        def _recolor_icon(image, color):
            image_data = np.asarray(image).copy()
            black_mask = (image_data[..., :3] == 0).all(axis=-1)
            image_data[black_mask, :3] = color
            return Image.fromarray(image_data, mode='RGBA')
        with Image.open(path_large if size > 24 and path_large.exists() else path_regular) as im:
            im = im.convert('RGBA')
            image = ImageTk.PhotoImage(im.resize((size, size)), master=self)
            button._ntimage = image
            foreground = 255 / 65535 * np.array(button.winfo_rgb(button.cget('foreground')))
            im_alt = _recolor_icon(im, foreground)
            image_alt = ImageTk.PhotoImage(im_alt.resize((size, size)), master=self)
            button._ntimage_alt = image_alt
        if _is_dark('background'):
            image_kwargs = {'image': image_alt}
        else:
            image_kwargs = {'image': image}
        if isinstance(button, tk.Checkbutton) and button.cget('selectcolor') != '':
            if self._windowingsystem != 'x11':
                selectcolor = 'selectcolor'
            else:
                r1, g1, b1 = _get_color('selectcolor')
                r2, g2, b2 = _get_color('activebackground')
                selectcolor = ((r1 + r2) / 2, (g1 + g2) / 2, (b1 + b2) / 2)
            if _is_dark(selectcolor):
                image_kwargs['selectimage'] = image_alt
            else:
                image_kwargs['selectimage'] = image
        button.configure(**image_kwargs, height='18p', width='18p')

    def _Button(self, text, image_file, toggle, command):
        if not toggle:
            b = tk.Button(master=self, text=text, command=command, relief='flat', overrelief='groove', borderwidth=1)
        else:
            var = tk.IntVar(master=self)
            b = tk.Checkbutton(master=self, text=text, command=command, indicatoron=False, variable=var, offrelief='flat', overrelief='groove', borderwidth=1)
            b.var = var
        b._image_file = image_file
        if image_file is not None:
            NavigationToolbar2Tk._set_image_for_button(self, b)
        else:
            b.configure(font=self._label_font)
        b.pack(side=tk.LEFT)
        return b

    def _Spacer(self):
        s = tk.Frame(master=self, height='18p', relief=tk.RIDGE, bg='DarkGray')
        s.pack(side=tk.LEFT, padx='3p')
        return s

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        tk_filetypes = [(name, ' '.join((f'*.{ext}' for ext in exts))) for name, exts in sorted(filetypes.items())]
        default_extension = self.canvas.get_default_filetype()
        default_filetype = self.canvas.get_supported_filetypes()[default_extension]
        filetype_variable = tk.StringVar(self, default_filetype)
        defaultextension = ''
        initialdir = os.path.expanduser(mpl.rcParams['savefig.directory'])
        initialfile = pathlib.Path(self.canvas.get_default_filename()).stem
        fname = tkinter.filedialog.asksaveasfilename(master=self.canvas.get_tk_widget().master, title='Save the figure', filetypes=tk_filetypes, defaultextension=defaultextension, initialdir=initialdir, initialfile=initialfile, typevariable=filetype_variable)
        if fname in ['', ()]:
            return
        if initialdir != '':
            mpl.rcParams['savefig.directory'] = os.path.dirname(str(fname))
        if pathlib.Path(fname).suffix[1:] != '':
            extension = None
        else:
            extension = filetypes[filetype_variable.get()][0]
        try:
            self.canvas.figure.savefig(fname, format=extension)
        except Exception as e:
            tkinter.messagebox.showerror('Error saving file', str(e))

    def set_history_buttons(self):
        state_map = {True: tk.NORMAL, False: tk.DISABLED}
        can_back = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        if 'Back' in self._buttons:
            self._buttons['Back']['state'] = state_map[can_back]
        if 'Forward' in self._buttons:
            self._buttons['Forward']['state'] = state_map[can_forward]