from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def __press_cb(self, event):
    """
        Handle a button-press event:
          - record the button press event in ``self.__press``
          - register a button-release callback.
          - if this CanvasWidget or any of its ancestors are
            draggable, then register the appropriate motion callback.
        """
    if self.__canvas.bind('<ButtonRelease-1>') or self.__canvas.bind('<ButtonRelease-2>') or self.__canvas.bind('<ButtonRelease-3>'):
        return
    self.__canvas.unbind('<Motion>')
    self.__press = event
    if event.num == 1:
        widget = self
        while widget is not None:
            if widget['draggable']:
                widget.__start_drag(event)
                break
            widget = widget.parent()
    self.__canvas.bind('<ButtonRelease-%d>' % event.num, self.__release_cb)