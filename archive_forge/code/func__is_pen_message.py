import os
from kivy.input.providers.wm_common import RECT, PEN_OR_TOUCH_MASK, \
from kivy.input.motionevent import MotionEvent
def _is_pen_message(self, msg):
    info = windll.user32.GetMessageExtraInfo()
    if info & PEN_OR_TOUCH_MASK == PEN_OR_TOUCH_SIGNATURE:
        if not info & PEN_EVENT_TOUCH_MASK:
            return True