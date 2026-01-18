from collections.abc import Iterable
import warnings
from typing import Sequence
def _text_dims(self, text_obj):
    """Get width and height of text object in data coordinates.

        See `this tutorial <https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html>`_
        for details on matplotlib coordinate systems.

        If the renderered figure is resized, such as in a GUI display, rectangles and lines
        are resized, but text stays the same size.  Text objects rely on display coordinates, that wont shrink
        as the figure is modified.

        Args:
            text_obj (matplotlib.text.Text): the matplotlib text object

        Returns:
            width (float): the width of the text in data coordinates
            height (float): the height of the text in data coordinates
        """
    renderer = self._fig.canvas.get_renderer()
    bbox = text_obj.get_window_extent(renderer)
    corners = self._ax.transData.inverted().transform(bbox)
    return (abs(corners[1][0] - corners[0][0]), abs(corners[0][1] - corners[1][1]))