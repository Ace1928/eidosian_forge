import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import _Style, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform
class AxislineStyle(_Style):
    """
    A container class which defines style classes for AxisArtists.

    An instance of any axisline style class is a callable object,
    whose call signature is ::

       __call__(self, axis_artist, path, transform)

    When called, this should return an `.Artist` with the following methods::

      def set_path(self, path):
          # set the path for axisline.

      def set_line_mutation_scale(self, scale):
          # set the scale

      def draw(self, renderer):
          # draw
    """
    _style_list = {}

    class _Base:

        def __init__(self):
            """
            initialization.
            """
            super().__init__()

        def __call__(self, axis_artist, transform):
            """
            Given the AxisArtist instance, and transform for the path (set_path
            method), return the Matplotlib artist for drawing the axis line.
            """
            return self.new_line(axis_artist, transform)

    class SimpleArrow(_Base):
        """
        A simple arrow.
        """
        ArrowAxisClass = _FancyAxislineStyle.SimpleArrow

        def __init__(self, size=1):
            """
            Parameters
            ----------
            size : float
                Size of the arrow as a fraction of the ticklabel size.
            """
            self.size = size
            super().__init__()

        def new_line(self, axis_artist, transform):
            linepath = Path([(0, 0), (0, 1)])
            axisline = self.ArrowAxisClass(axis_artist, linepath, transform, line_mutation_scale=self.size)
            return axisline
    _style_list['->'] = SimpleArrow

    class FilledArrow(SimpleArrow):
        """
        An arrow with a filled head.
        """
        ArrowAxisClass = _FancyAxislineStyle.FilledArrow

        def __init__(self, size=1, facecolor=None):
            """
            Parameters
            ----------
            size : float
                Size of the arrow as a fraction of the ticklabel size.
            facecolor : color, default: :rc:`axes.edgecolor`
                Fill color.

                .. versionadded:: 3.7
            """
            if facecolor is None:
                facecolor = mpl.rcParams['axes.edgecolor']
            self.size = size
            self._facecolor = facecolor
            super().__init__(size=size)

        def new_line(self, axis_artist, transform):
            linepath = Path([(0, 0), (0, 1)])
            axisline = self.ArrowAxisClass(axis_artist, linepath, transform, line_mutation_scale=self.size, facecolor=self._facecolor)
            return axisline
    _style_list['-|>'] = FilledArrow