from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def __motion_cb(self, event):
    """
        Handle a motion event:
          - move this object to the new location
          - record the new drag coordinates
        """
    self.move(event.x - self.__drag_x, event.y - self.__drag_y)
    self.__drag_x = event.x
    self.__drag_y = event.y