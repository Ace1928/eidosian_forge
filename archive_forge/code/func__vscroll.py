import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
def _vscroll(clicks, x, y):
    """A wrapper for _scroll(), which does vertical scrolling.

    Args:
      clicks (int): The amount of scrolling to do. A positive value is the mouse
      wheel moving forward (scrolling up), a negative value is backwards (down).
      x (int): The x position of the mouse event.
      y (int): The y position of the mouse event.

    Returns:
      None
    """
    return _scroll(clicks, x, y)