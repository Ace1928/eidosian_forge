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