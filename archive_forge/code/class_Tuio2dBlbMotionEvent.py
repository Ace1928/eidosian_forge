from kivy.logger import Logger
from functools import partial
from collections import deque
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class Tuio2dBlbMotionEvent(TuioMotionEvent):
    """A 2dBlb TUIO object.
    # FIXME 3d shape are not supported
    /tuio/2Dobj set s i x y a       X Y A m r
    /tuio/2Dblb set s   x y a w h f X Y A m r
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = ('pos', 'angle', 'mov', 'rot', 'rotacc', 'acc', 'shape')

    def depack(self, args):
        self.sx, self.sy, self.a, self.X, self.Y, sw, sh, sd, self.A, self.m, self.r = args
        self.Y = -self.Y
        if self.shape is None:
            self.shape = ShapeRect()
            self.shape.width = sw
            self.shape.height = sh
        self.sy = 1 - self.sy
        super().depack(args)