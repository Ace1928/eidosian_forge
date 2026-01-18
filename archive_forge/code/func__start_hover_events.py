from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def _start_hover_events(self):
    fbind = EventLoop.window.fbind
    fbind('mouse_pos', self.begin_or_update_hover_event)
    fbind('system_size', self.update_hover_event)
    fbind('on_cursor_enter', self.begin_hover_event)
    fbind('on_cursor_leave', self.end_hover_event)
    fbind('on_close', self.end_hover_event)
    fbind('on_rotate', self.update_hover_event)