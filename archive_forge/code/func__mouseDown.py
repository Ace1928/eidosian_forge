import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
def _mouseDown(x, y, button):
    """Send the mouse down event to Windows by calling the mouse_event() win32
    function.

    Args:
      x (int): The x position of the mouse event.
      y (int): The y position of the mouse event.
      button (str): The mouse button, either 'left', 'middle', or 'right'

    Returns:
      None
    """
    if button not in (LEFT, MIDDLE, RIGHT):
        raise ValueError('button arg to _click() must be one of "left", "middle", or "right", not %s' % button)
    if button == LEFT:
        EV = MOUSEEVENTF_LEFTDOWN
    elif button == MIDDLE:
        EV = MOUSEEVENTF_MIDDLEDOWN
    elif button == RIGHT:
        EV = MOUSEEVENTF_RIGHTDOWN
    try:
        _sendMouseEvent(EV, x, y)
    except (PermissionError, OSError):
        pass