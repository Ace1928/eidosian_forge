import os
from kivy.input.providers.wm_common import WNDPROC, \
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def _touch_handler(self, msg, wParam, lParam):
    touches = (TOUCHINPUT * wParam)()
    windll.user32.GetTouchInputInfo(HANDLE(lParam), wParam, touches, sizeof(TOUCHINPUT))
    for i in range(wParam):
        self.touch_events.appendleft(touches[i])
    windll.user32.CloseTouchInputHandle(HANDLE(lParam))
    return True