from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class OvalWidget(AbstractContainerWidget):
    """
    A canvas widget that places a oval around a child widget.

    Attributes:
      - ``fill``: The color used to fill the interior of the oval.
      - ``outline``: The color used to draw the outline of the oval.
      - ``width``: The width of the outline of the oval.
      - ``margin``: The number of pixels space left between the child
        and the oval.
      - ``draggable``: whether the text can be dragged by the user.
      - ``double``: If true, then a double-oval is drawn.
    """

    def __init__(self, canvas, child, **attribs):
        """
        Create a new oval widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :param child: The child widget.  ``child`` must not have a
            parent.
        :type child: CanvasWidget
        :param attribs: The new canvas widget's attributes.
        """
        self._child = child
        self._margin = 1
        self._oval = canvas.create_oval(1, 1, 1, 1)
        self._circle = attribs.pop('circle', False)
        self._double = attribs.pop('double', False)
        if self._double:
            self._oval2 = canvas.create_oval(1, 1, 1, 1)
        else:
            self._oval2 = None
        canvas.tag_lower(self._oval)
        AbstractContainerWidget.__init__(self, canvas, child, **attribs)

    def __setitem__(self, attr, value):
        c = self.canvas()
        if attr == 'margin':
            self._margin = value
        elif attr == 'double':
            if value == True and self._oval2 is None:
                x1, y1, x2, y2 = c.bbox(self._oval)
                w = self['width'] * 2
                self._oval2 = c.create_oval(x1 - w, y1 - w, x2 + w, y2 + w, outline=c.itemcget(self._oval, 'outline'), width=c.itemcget(self._oval, 'width'))
                c.tag_lower(self._oval2)
            if value == False and self._oval2 is not None:
                c.delete(self._oval2)
                self._oval2 = None
        elif attr in ('outline', 'fill', 'width'):
            c.itemconfig(self._oval, {attr: value})
            if self._oval2 is not None and attr != 'fill':
                c.itemconfig(self._oval2, {attr: value})
            if self._oval2 is not None and attr != 'fill':
                self.canvas().itemconfig(self._oval2, {attr: value})
        else:
            CanvasWidget.__setitem__(self, attr, value)

    def __getitem__(self, attr):
        if attr == 'margin':
            return self._margin
        elif attr == 'double':
            return self._double is not None
        elif attr == 'width':
            return float(self.canvas().itemcget(self._oval, attr))
        elif attr in ('outline', 'fill', 'width'):
            return self.canvas().itemcget(self._oval, attr)
        else:
            return CanvasWidget.__getitem__(self, attr)
    RATIO = 1.414213562373095

    def _update(self, child):
        R = OvalWidget.RATIO
        x1, y1, x2, y2 = child.bbox()
        margin = self._margin
        if self._circle:
            dx, dy = (abs(x1 - x2), abs(y1 - y2))
            if dx > dy:
                y = (y1 + y2) / 2
                y1, y2 = (y - dx / 2, y + dx / 2)
            elif dy > dx:
                x = (x1 + x2) / 2
                x1, x2 = (x - dy / 2, x + dy / 2)
        left = int((x1 * (1 + R) + x2 * (1 - R)) / 2)
        right = left + int((x2 - x1) * R)
        top = int((y1 * (1 + R) + y2 * (1 - R)) / 2)
        bot = top + int((y2 - y1) * R)
        self.canvas().coords(self._oval, left - margin, top - margin, right + margin, bot + margin)
        if self._oval2 is not None:
            self.canvas().coords(self._oval2, left - margin + 2, top - margin + 2, right + margin - 2, bot + margin - 2)

    def _tags(self):
        if self._oval2 is None:
            return [self._oval]
        else:
            return [self._oval, self._oval2]