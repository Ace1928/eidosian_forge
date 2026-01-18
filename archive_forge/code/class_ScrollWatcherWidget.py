from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class ScrollWatcherWidget(CanvasWidget):
    """
    A special canvas widget that adjusts its ``Canvas``'s scrollregion
    to always include the bounding boxes of all of its children.  The
    scroll-watcher widget will only increase the size of the
    ``Canvas``'s scrollregion; it will never decrease it.
    """

    def __init__(self, canvas, *children, **attribs):
        """
        Create a new scroll-watcher widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :type children: list(CanvasWidget)
        :param children: The canvas widgets watched by the
            scroll-watcher.  The scroll-watcher will ensure that these
            canvas widgets are always contained in their canvas's
            scrollregion.
        :param attribs: The new canvas widget's attributes.
        """
        for child in children:
            self._add_child_widget(child)
        CanvasWidget.__init__(self, canvas, **attribs)

    def add_child(self, canvaswidget):
        """
        Add a new canvas widget to the scroll-watcher.  The
        scroll-watcher will ensure that the new canvas widget is
        always contained in its canvas's scrollregion.

        :param canvaswidget: The new canvas widget.
        :type canvaswidget: CanvasWidget
        :rtype: None
        """
        self._add_child_widget(canvaswidget)
        self.update(canvaswidget)

    def remove_child(self, canvaswidget):
        """
        Remove a canvas widget from the scroll-watcher.  The
        scroll-watcher will no longer ensure that the new canvas
        widget is always contained in its canvas's scrollregion.

        :param canvaswidget: The canvas widget to remove.
        :type canvaswidget: CanvasWidget
        :rtype: None
        """
        self._remove_child_widget(canvaswidget)

    def _tags(self):
        return []

    def _update(self, child):
        self._adjust_scrollregion()

    def _adjust_scrollregion(self):
        """
        Adjust the scrollregion of this scroll-watcher's ``Canvas`` to
        include the bounding boxes of all of its children.
        """
        bbox = self.bbox()
        canvas = self.canvas()
        scrollregion = [int(n) for n in canvas['scrollregion'].split()]
        if len(scrollregion) != 4:
            return
        if bbox[0] < scrollregion[0] or bbox[1] < scrollregion[1] or bbox[2] > scrollregion[2] or (bbox[3] > scrollregion[3]):
            scrollregion = '%d %d %d %d' % (min(bbox[0], scrollregion[0]), min(bbox[1], scrollregion[1]), max(bbox[2], scrollregion[2]), max(bbox[3], scrollregion[3]))
            canvas['scrollregion'] = scrollregion