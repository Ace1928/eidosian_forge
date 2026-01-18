from kivy.logger import Logger
from functools import partial
from collections import deque
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class Tuio2dCurMotionEvent(TuioMotionEvent):
    """A 2dCur TUIO touch."""

    def depack(self, args):
        if len(args) < 5:
            self.sx, self.sy = list(map(float, args[0:2]))
            self.profile = ('pos',)
        elif len(args) == 5:
            self.sx, self.sy, self.X, self.Y, self.m = list(map(float, args[0:5]))
            self.Y = -self.Y
            self.profile = ('pos', 'mov', 'motacc')
        else:
            self.sx, self.sy, self.X, self.Y = list(map(float, args[0:4]))
            self.m, width, height = list(map(float, args[4:7]))
            self.Y = -self.Y
            self.profile = ('pos', 'mov', 'motacc', 'shape')
            if self.shape is None:
                self.shape = ShapeRect()
            self.shape.width = width
            self.shape.height = height
        self.sy = 1 - self.sy
        super().depack(args)