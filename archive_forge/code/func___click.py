from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def __click(self, button):
    """
        If this ``CanvasWidget`` has a drag callback, then call it;
        otherwise, find the closest ancestor with a click callback, and
        call it.  If no ancestors have a click callback, do nothing.
        """
    if button in self.__callbacks:
        cb = self.__callbacks[button]
        cb(self)
    elif self.__parent is not None:
        self.__parent.__click(button)