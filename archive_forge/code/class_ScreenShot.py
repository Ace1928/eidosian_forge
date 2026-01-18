from threading import RLock
from sympy.core.numbers import Integer
from sympy.external.gmpy import SYMPY_INTS
from sympy.geometry.entity import GeometryEntity
from sympy.plotting.pygletplot.plot_axes import PlotAxes
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.plot_window import PlotWindow
from sympy.plotting.pygletplot.util import parse_option_string
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import is_sequence
from time import sleep
from os import getcwd, listdir
import ctypes
class ScreenShot:

    def __init__(self, plot):
        self._plot = plot
        self.screenshot_requested = False
        self.outfile = None
        self.format = ''
        self.invisibleMode = False
        self.flag = 0

    def __bool__(self):
        return self.screenshot_requested

    def _execute_saving(self):
        if self.flag < 3:
            self.flag += 1
            return
        size_x, size_y = self._plot._window.get_size()
        size = size_x * size_y * 4 * ctypes.sizeof(ctypes.c_ubyte)
        image = ctypes.create_string_buffer(size)
        pgl.glReadPixels(0, 0, size_x, size_y, pgl.GL_RGBA, pgl.GL_UNSIGNED_BYTE, image)
        from PIL import Image
        im = Image.frombuffer('RGBA', (size_x, size_y), image.raw, 'raw', 'RGBA', 0, 1)
        im.transpose(Image.FLIP_TOP_BOTTOM).save(self.outfile, self.format)
        self.flag = 0
        self.screenshot_requested = False
        if self.invisibleMode:
            self._plot._window.close()

    def save(self, outfile=None, format='', size=(600, 500)):
        self.outfile = outfile
        self.format = format
        self.size = size
        self.screenshot_requested = True
        if not self._plot._window or self._plot._window.has_exit:
            self._plot._win_args['visible'] = False
            self._plot._win_args['width'] = size[0]
            self._plot._win_args['height'] = size[1]
            self._plot.axes.reset_resources()
            self._plot._window = PlotWindow(self._plot, **self._plot._win_args)
            self.invisibleMode = True
        if self.outfile is None:
            self.outfile = self._create_unique_path()
            print(self.outfile)

    def _create_unique_path(self):
        cwd = getcwd()
        l = listdir(cwd)
        path = ''
        i = 0
        while True:
            if not 'plot_%s.png' % i in l:
                path = cwd + '/plot_%s.png' % i
                break
            i += 1
        return path