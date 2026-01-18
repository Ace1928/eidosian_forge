from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class StackWidget(CanvasWidget):
    """
    A canvas widget that keeps a list of canvas widgets in a vertical
    line.

    Attributes:
      - ``align``: The horizontal alignment of the children.  Possible
        values are ``'left'``, ``'center'``, and ``'right'``.  By
        default, children are center-aligned.
      - ``space``: The amount of vertical space to place between
        children.  By default, one pixel of space is used.
      - ``ordered``: If true, then keep the children in their
        original order.
    """

    def __init__(self, canvas, *children, **attribs):
        """
        Create a new stack widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :param children: The widgets that should be aligned
            vertically.  Each child must not have a parent.
        :type children: list(CanvasWidget)
        :param attribs: The new canvas widget's attributes.
        """
        self._align = 'center'
        self._space = 1
        self._ordered = False
        self._children = list(children)
        for child in children:
            self._add_child_widget(child)
        CanvasWidget.__init__(self, canvas, **attribs)

    def __setitem__(self, attr, value):
        if attr == 'align':
            if value not in ('left', 'right', 'center'):
                raise ValueError('Bad alignment: %r' % value)
            self._align = value
        elif attr == 'space':
            self._space = value
        elif attr == 'ordered':
            self._ordered = value
        else:
            CanvasWidget.__setitem__(self, attr, value)

    def __getitem__(self, attr):
        if attr == 'align':
            return self._align
        elif attr == 'space':
            return self._space
        elif attr == 'ordered':
            return self._ordered
        else:
            return CanvasWidget.__getitem__(self, attr)

    def _tags(self):
        return []

    def _xalign(self, left, right):
        if self._align == 'left':
            return left
        if self._align == 'right':
            return right
        if self._align == 'center':
            return (left + right) / 2

    def _update(self, child):
        left, top, right, bot = child.bbox()
        x = self._xalign(left, right)
        for c in self._children:
            x1, y1, x2, y2 = c.bbox()
            c.move(x - self._xalign(x1, x2), 0)
        if self._ordered and len(self._children) > 1:
            index = self._children.index(child)
            y = bot + self._space
            for i in range(index + 1, len(self._children)):
                x1, y1, x2, y2 = self._children[i].bbox()
                if y > y1:
                    self._children[i].move(0, y - y1)
                    y += y2 - y1 + self._space
            y = top - self._space
            for i in range(index - 1, -1, -1):
                x1, y1, x2, y2 = self._children[i].bbox()
                if y < y2:
                    self._children[i].move(0, y - y2)
                    y -= y2 - y1 + self._space

    def _manage(self):
        if len(self._children) == 0:
            return
        child = self._children[0]
        left, top, right, bot = child.bbox()
        x = self._xalign(left, right)
        index = self._children.index(child)
        y = bot + self._space
        for i in range(index + 1, len(self._children)):
            x1, y1, x2, y2 = self._children[i].bbox()
            self._children[i].move(x - self._xalign(x1, x2), y - y1)
            y += y2 - y1 + self._space
        y = top - self._space
        for i in range(index - 1, -1, -1):
            x1, y1, x2, y2 = self._children[i].bbox()
            self._children[i].move(x - self._xalign(x1, x2), y - y2)
            y -= y2 - y1 + self._space

    def __repr__(self):
        return '[Stack: ' + repr(self._children)[1:-1] + ']'
    children = CanvasWidget.child_widgets

    def replace_child(self, oldchild, newchild):
        """
        Replace the child canvas widget ``oldchild`` with ``newchild``.
        ``newchild`` must not have a parent.  ``oldchild``'s parent will
        be set to None.

        :type oldchild: CanvasWidget
        :param oldchild: The child canvas widget to remove.
        :type newchild: CanvasWidget
        :param newchild: The canvas widget that should replace
            ``oldchild``.
        """
        index = self._children.index(oldchild)
        self._children[index] = newchild
        self._remove_child_widget(oldchild)
        self._add_child_widget(newchild)
        self.update(newchild)

    def remove_child(self, child):
        """
        Remove the given child canvas widget.  ``child``'s parent will
        be set to None.

        :type child: CanvasWidget
        :param child: The child canvas widget to remove.
        """
        index = self._children.index(child)
        del self._children[index]
        self._remove_child_widget(child)
        if len(self._children) > 0:
            self.update(self._children[0])

    def insert_child(self, index, child):
        """
        Insert a child canvas widget before a given index.

        :type child: CanvasWidget
        :param child: The canvas widget that should be inserted.
        :type index: int
        :param index: The index where the child widget should be
            inserted.  In particular, the index of ``child`` will be
            ``index``; and the index of any children whose indices were
            greater than equal to ``index`` before ``child`` was
            inserted will be incremented by one.
        """
        self._children.insert(index, child)
        self._add_child_widget(child)