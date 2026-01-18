from __future__ import annotations
from . import Image
class HWND:
    """
    Wraps an HWND integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods, instead of a DC.
    """

    def __init__(self, wnd):
        self.wnd = wnd

    def __int__(self):
        return self.wnd