import time
import sys
import AppKit
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
def _multiClick(x, y, button, num, interval=0.0):
    btn = None
    down = None
    up = None
    if button == LEFT:
        btn = Quartz.kCGMouseButtonLeft
        down = Quartz.kCGEventLeftMouseDown
        up = Quartz.kCGEventLeftMouseUp
    elif button == MIDDLE:
        btn = Quartz.kCGMouseButtonCenter
        down = Quartz.kCGEventOtherMouseDown
        up = Quartz.kCGEventOtherMouseUp
    elif button == RIGHT:
        btn = Quartz.kCGMouseButtonRight
        down = Quartz.kCGEventRightMouseDown
        up = Quartz.kCGEventRightMouseUp
    else:
        assert False, "button argument not in ('left', 'middle', 'right')"
        return
    for i in range(num):
        _click(x, y, button)
        time.sleep(interval)