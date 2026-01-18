import tkFont
import Tkinter
import rdkit.sping.pid
def _getTkColor(self, color, defaultColor):
    if color is None:
        color = defaultColor
    if color is rdkit.sping.pid.transparent:
        color = self.__TRANSPARENT
    else:
        color = self._colorToTkColor(color)
    return color