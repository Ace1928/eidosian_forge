import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class LinuxWacomMotionEvent(MotionEvent):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)

    def depack(self, args):
        self.sx = args['x']
        self.sy = args['y']
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
        return '<LinuxWacomMotionEvent id=%d pos=(%f, %f) device=%s>' % (self.id, self.sx, self.sy, self.device)