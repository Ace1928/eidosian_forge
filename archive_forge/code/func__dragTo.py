import time
import sys
import AppKit
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
def _dragTo(x, y, button):
    if button == LEFT:
        _sendMouseEvent(Quartz.kCGEventLeftMouseDragged, x, y, Quartz.kCGMouseButtonLeft)
    elif button == MIDDLE:
        _sendMouseEvent(Quartz.kCGEventOtherMouseDragged, x, y, Quartz.kCGMouseButtonCenter)
    elif button == RIGHT:
        _sendMouseEvent(Quartz.kCGEventRightMouseDragged, x, y, Quartz.kCGMouseButtonRight)
    else:
        assert False, "button argument not in ('left', 'middle', 'right')"
    time.sleep(pyautogui.DARWIN_CATCH_UP_TIME)