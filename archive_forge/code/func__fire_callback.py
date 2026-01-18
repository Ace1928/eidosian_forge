from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _fire_callback(self, event, itemnum):
    if event not in self._callbacks:
        return
    if 0 <= itemnum < len(self._items):
        item = self._items[itemnum]
    else:
        item = None
    for cb_func in list(self._callbacks[event].keys()):
        cb_func(item)