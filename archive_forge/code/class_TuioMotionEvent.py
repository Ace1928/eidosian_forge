from kivy.logger import Logger
from functools import partial
from collections import deque
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class TuioMotionEvent(MotionEvent):
    """Abstraction for TUIO touches/fiducials.

    Depending on the tracking software you use (e.g. Movid, CCV, etc.) and its
    TUIO implementation, the TuioMotionEvent object can support multiple
    profiles such as:

        * Fiducial ID: profile name 'markerid', attribute ``.fid``
        * Position: profile name 'pos', attributes ``.x``, ``.y``
        * Angle: profile name 'angle', attribute ``.a``
        * Velocity vector: profile name 'mov', attributes ``.X``, ``.Y``
        * Rotation velocity: profile name 'rot', attribute ``.A``
        * Motion acceleration: profile name 'motacc', attribute ``.m``
        * Rotation acceleration: profile name 'rotacc', attribute ``.r``
    """
    __attrs__ = ('a', 'b', 'c', 'X', 'Y', 'Z', 'A', 'B', 'C', 'm', 'r')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.A = 0.0
        self.B = 0.0
        self.C = 0.0
        self.m = 0.0
        self.r = 0.0
    angle = property(lambda self: self.a)
    mot_accel = property(lambda self: self.m)
    rot_accel = property(lambda self: self.r)
    xmot = property(lambda self: self.X)
    ymot = property(lambda self: self.Y)
    zmot = property(lambda self: self.Z)