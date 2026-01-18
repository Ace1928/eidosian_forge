from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class SpaceWidget(CanvasWidget):
    """
    A canvas widget that takes up space but does not display
    anything.  A ``SpaceWidget`` can be used to add space between
    elements.  Each space widget is characterized by a width and a
    height.  If you wish to only create horizontal space, then use a
    height of zero; and if you wish to only create vertical space, use
    a width of zero.
    """

    def __init__(self, canvas, width, height, **attribs):
        """
        Create a new space widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :type width: int
        :param width: The width of the new space widget.
        :type height: int
        :param height: The height of the new space widget.
        :param attribs: The new canvas widget's attributes.
        """
        if width > 4:
            width -= 4
        if height > 4:
            height -= 4
        self._tag = canvas.create_line(1, 1, width, height, fill='')
        CanvasWidget.__init__(self, canvas, **attribs)

    def set_width(self, width):
        """
        Change the width of this space widget.

        :param width: The new width.
        :type width: int
        :rtype: None
        """
        [x1, y1, x2, y2] = self.bbox()
        self.canvas().coords(self._tag, x1, y1, x1 + width, y2)

    def set_height(self, height):
        """
        Change the height of this space widget.

        :param height: The new height.
        :type height: int
        :rtype: None
        """
        [x1, y1, x2, y2] = self.bbox()
        self.canvas().coords(self._tag, x1, y1, x2, y1 + height)

    def _tags(self):
        return [self._tag]

    def __repr__(self):
        return '[Space]'