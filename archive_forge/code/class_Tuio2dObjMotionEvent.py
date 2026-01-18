from kivy.logger import Logger
from functools import partial
from collections import deque
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class Tuio2dObjMotionEvent(TuioMotionEvent):
    """A 2dObj TUIO object.
    """

    def depack(self, args):
        if len(args) < 5:
            self.sx, self.sy = args[0:2]
            self.profile = ('pos',)
        elif len(args) == 9:
            self.fid, self.sx, self.sy, self.a, self.X, self.Y = args[:6]
            self.A, self.m, self.r = args[6:9]
            self.Y = -self.Y
            self.profile = ('markerid', 'pos', 'angle', 'mov', 'rot', 'motacc', 'rotacc')
        else:
            self.fid, self.sx, self.sy, self.a, self.X, self.Y = args[:6]
            self.A, self.m, self.r, width, height = args[6:11]
            self.Y = -self.Y
            self.profile = ('markerid', 'pos', 'angle', 'mov', 'rot', 'rotacc', 'acc', 'shape')
            if self.shape is None:
                self.shape = ShapeRect()
                self.shape.width = width
                self.shape.height = height
        self.sy = 1 - self.sy
        super().depack(args)