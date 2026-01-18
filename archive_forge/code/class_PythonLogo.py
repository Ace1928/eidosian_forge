from __future__ import annotations
from urwid.display import AttrSpec
from urwid.widget import (
class PythonLogo(Widget):
    _sizing = frozenset([Sizing.FIXED])

    def __init__(self) -> None:
        """
        Create canvas containing an ASCII version of the Python
        Logo and store it.
        """
        super().__init__()
        blu = AttrSpec('light blue', 'default')
        yel = AttrSpec('yellow', 'default')
        width = 17
        self._canvas = Text([(blu, '     ______\n'), (blu, '   _|_o__  |'), (yel, '__\n'), (blu, '  |   _____|'), (yel, '  |\n'), (blu, '  |__|  '), (yel, '______|\n'), (yel, '     |____o_|')]).render((width,))

    def pack(self, size=None, focus: bool=False):
        """
        Return the size from our pre-rendered canvas.
        """
        return (self._canvas.cols(), self._canvas.rows())

    def render(self, size, focus: bool=False):
        """
        Return the pre-rendered canvas.
        """
        fixed_size(size)
        return self._canvas