from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def __drag(self):
    """
        If this ``CanvasWidget`` has a drag callback, then call it;
        otherwise, find the closest ancestor with a drag callback, and
        call it.  If no ancestors have a drag callback, do nothing.
        """
    if self.__draggable:
        if 'drag' in self.__callbacks:
            cb = self.__callbacks['drag']
            try:
                cb(self)
            except:
                print('Error in drag callback for %r' % self)
    elif self.__parent is not None:
        self.__parent.__drag()