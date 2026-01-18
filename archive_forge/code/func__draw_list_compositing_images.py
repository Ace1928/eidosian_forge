import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
def _draw_list_compositing_images(renderer, parent, artists, suppress_composite=None):
    """
    Draw a sorted list of artists, compositing images into a single
    image where possible.

    For internal Matplotlib use only: It is here to reduce duplication
    between `Figure.draw` and `Axes.draw`, but otherwise should not be
    generally useful.
    """
    has_images = any((isinstance(x, _ImageBase) for x in artists))
    not_composite = suppress_composite if suppress_composite is not None else renderer.option_image_nocomposite()
    if not_composite or not has_images:
        for a in artists:
            a.draw(renderer)
    else:
        image_group = []
        mag = renderer.get_image_magnification()

        def flush_images():
            if len(image_group) == 1:
                image_group[0].draw(renderer)
            elif len(image_group) > 1:
                data, l, b = composite_images(image_group, renderer, mag)
                if data.size != 0:
                    gc = renderer.new_gc()
                    gc.set_clip_rectangle(parent.bbox)
                    gc.set_clip_path(parent.get_clip_path())
                    renderer.draw_image(gc, round(l), round(b), data)
                    gc.restore()
            del image_group[:]
        for a in artists:
            if isinstance(a, _ImageBase) and a.can_composite() and a.get_clip_on() and (not a.get_clip_path()):
                image_group.append(a)
            else:
                flush_images()
                a.draw(renderer)
        flush_images()