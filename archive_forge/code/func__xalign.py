from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _xalign(self, left, right):
    if self._align == 'left':
        return left
    if self._align == 'right':
        return right
    if self._align == 'center':
        return (left + right) / 2