from typing import Callable
from fontTools.pens.basePen import BasePen
def _handleAnchor(self):
    """
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen.moveTo((10, 10))
        >>> pen._commands
        ['M10 10']
        """
    if self._lastCommand == 'M':
        self._commands.pop(-1)