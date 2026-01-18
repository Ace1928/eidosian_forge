from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class ParenWidget(AbstractContainerWidget):
    """
    A canvas widget that places a pair of parenthases around a child
    widget.

    Attributes:
      - ``color``: The color used to draw the parenthases.
      - ``width``: The width of the parenthases.
      - ``draggable``: whether the text can be dragged by the user.
    """

    def __init__(self, canvas, child, **attribs):
        """
        Create a new parenthasis widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :param child: The child widget.  ``child`` must not have a
            parent.
        :type child: CanvasWidget
        :param attribs: The new canvas widget's attributes.
        """
        self._child = child
        self._oparen = canvas.create_arc(1, 1, 1, 1, style='arc', start=90, extent=180)
        self._cparen = canvas.create_arc(1, 1, 1, 1, style='arc', start=-90, extent=180)
        AbstractContainerWidget.__init__(self, canvas, child, **attribs)

    def __setitem__(self, attr, value):
        if attr == 'color':
            self.canvas().itemconfig(self._oparen, outline=value)
            self.canvas().itemconfig(self._cparen, outline=value)
        elif attr == 'width':
            self.canvas().itemconfig(self._oparen, width=value)
            self.canvas().itemconfig(self._cparen, width=value)
        else:
            CanvasWidget.__setitem__(self, attr, value)

    def __getitem__(self, attr):
        if attr == 'color':
            return self.canvas().itemcget(self._oparen, 'outline')
        elif attr == 'width':
            return self.canvas().itemcget(self._oparen, 'width')
        else:
            return CanvasWidget.__getitem__(self, attr)

    def _update(self, child):
        x1, y1, x2, y2 = child.bbox()
        width = max((y2 - y1) / 6, 4)
        self.canvas().coords(self._oparen, x1 - width, y1, x1 + width, y2)
        self.canvas().coords(self._cparen, x2 - width, y1, x2 + width, y2)

    def _tags(self):
        return [self._oparen, self._cparen]