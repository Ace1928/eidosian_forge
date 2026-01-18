import logging
from reportlab import rl_config
def _geom(self):
    self._x2 = self._x1 + self._width
    self._y2 = self._y1 + self._height
    self._y1p = self._y1 + self._bottomPadding
    self._aW = self._x2 - self._x1 - self._leftPadding - self._rightPadding
    self._aH = self._y2 - self._y1p - self._topPadding