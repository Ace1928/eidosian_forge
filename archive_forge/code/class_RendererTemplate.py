from matplotlib import _api
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.figure import Figure
class RendererTemplate(RendererBase):
    """
    The renderer handles drawing/rendering operations.

    This is a minimal do-nothing class that can be used to get started when
    writing a new backend.  Refer to `.backend_bases.RendererBase` for
    documentation of the methods.
    """

    def __init__(self, dpi):
        super().__init__()
        self.dpi = dpi

    def draw_path(self, gc, path, transform, rgbFace=None):
        pass

    def draw_image(self, gc, x, y, im):
        pass

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        pass

    def flipy(self):
        return True

    def get_canvas_width_height(self):
        return (100, 100)

    def get_text_width_height_descent(self, s, prop, ismath):
        return (1, 1, 1)

    def new_gc(self):
        return GraphicsContextTemplate()

    def points_to_pixels(self, points):
        return points