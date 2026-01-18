import copy
from math import *
from qt import *
from qtcanvas import *
from rdkit.sping import pid
def _adjustFont(self, font):
    if font.face:
        self._font.setFamily(font.face)
    self._font.setBold(font.bold)
    self._font.setItalic(font.italic)
    self._font.setPointSize(font.size)
    self._font.setUnderline(font.underline)