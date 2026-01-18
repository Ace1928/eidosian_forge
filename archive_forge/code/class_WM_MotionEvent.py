import os
from kivy.input.providers.wm_common import WNDPROC, \
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class WM_MotionEvent(MotionEvent):
    """MotionEvent representing the WM_MotionEvent event.
       Supports pos, shape and size profiles.
    """
    __attrs__ = ('size',)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)
        self.profile = ('pos', 'shape', 'size')

    def depack(self, args):
        self.shape = ShapeRect()
        self.sx, self.sy = (args[0], args[1])
        self.shape.width = args[2][0]
        self.shape.height = args[2][1]
        self.size = self.shape.width * self.shape.height
        super().depack(args)

    def __str__(self):
        args = (self.id, self.uid, str(self.spos), self.device)
        return '<WMMotionEvent id:%d uid:%d pos:%s device:%s>' % args