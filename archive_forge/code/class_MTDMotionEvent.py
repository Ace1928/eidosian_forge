import os
import os.path
import time
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class MTDMotionEvent(MotionEvent):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)

    def depack(self, args):
        if 'x' in args:
            self.sx = args['x']
        else:
            self.sx = -1
        if 'y' in args:
            self.sy = args['y']
        else:
            self.sy = -1
        self.profile = ['pos']
        if 'size_w' in args and 'size_h' in args:
            self.shape = ShapeRect()
            self.shape.width = args['size_w']
            self.shape.height = args['size_h']
            self.profile.append('shape')
        if 'pressure' in args:
            self.pressure = args['pressure']
            self.profile.append('pressure')
        super().depack(args)

    def __str__(self):
        i, sx, sy, d = (self.id, self.sx, self.sy, self.device)
        return '<MTDMotionEvent id=%d pos=(%f, %f) device=%s>' % (i, sx, sy, d)