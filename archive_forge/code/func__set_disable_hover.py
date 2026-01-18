from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def _set_disable_hover(self, value):
    if self._disable_hover != value:
        if self._running:
            if value:
                self._stop_hover_events()
            else:
                self._start_hover_events()
        self._disable_hover = value