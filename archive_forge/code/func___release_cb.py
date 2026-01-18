from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def __release_cb(self, event):
    """
        Handle a release callback:
          - unregister motion & button release callbacks.
          - decide whether they clicked, dragged, or cancelled
          - call the appropriate handler.
        """
    self.__canvas.unbind('<ButtonRelease-%d>' % event.num)
    self.__canvas.unbind('<Motion>')
    if event.time - self.__press.time < 100 and abs(event.x - self.__press.x) + abs(event.y - self.__press.y) < 5:
        if self.__draggable and event.num == 1:
            self.move(self.__press.x - self.__drag_x, self.__press.y - self.__drag_y)
        self.__click(event.num)
    elif event.num == 1:
        self.__drag()
    self.__press = None