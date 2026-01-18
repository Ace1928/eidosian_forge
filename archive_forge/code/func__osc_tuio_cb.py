from kivy.logger import Logger
from functools import partial
from collections import deque
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def _osc_tuio_cb(self, oscpath, address, *args):
    self.tuio_event_q.appendleft([oscpath, address, args])