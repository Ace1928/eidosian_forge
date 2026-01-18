import os
from kivy.input.providers.wm_common import RECT, PEN_OR_TOUCH_MASK, \
from kivy.input.motionevent import MotionEvent
class WM_Pen(MotionEvent):
    """MotionEvent representing the WM_Pen event. Supports the pos profile."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)
        self.profile = ['pos']

    def depack(self, args):
        self.sx, self.sy = (args[0], args[1])
        super().depack(args)

    def __str__(self):
        i, u, s, d = (self.id, self.uid, str(self.spos), self.device)
        return '<WMPen id:%d uid:%d pos:%s device:%s>' % (i, u, s, d)