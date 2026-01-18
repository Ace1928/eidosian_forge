from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
class MouseMotionEvent(MotionEvent):

    def __init__(self, *args, **kwargs):
        self.multitouch_sim = False
        super().__init__(*args, **kwargs)

    def depack(self, args):
        self.sx, self.sy = args[:2]
        profile = self.profile
        if self.is_touch:
            if not profile:
                profile.extend(('pos', 'button'))
            if len(args) >= 3:
                self.button = args[2]
            if len(args) == 4:
                self.multitouch_sim = args[3]
                profile.append('multitouch_sim')
        elif not profile:
            profile.append('pos')
        super().depack(args)

    def update_graphics(self, win, create=False):
        global Color, Ellipse
        de = self.ud.get('_drawelement', None)
        if de is None and create:
            if Color is None:
                from kivy.graphics import Color, Ellipse
            with win.canvas.after:
                de = (Color(0.8, 0.2, 0.2, 0.7), Ellipse(size=(20, 20), segments=15))
            self.ud._drawelement = de
        if de is not None:
            self.push()
            w, h = win._get_effective_size()
            self.scale_for_screen(w, h, rotation=win.rotation)
            de[1].pos = (self.x - 10, self.y - 10)
            self.pop()

    def clear_graphics(self, win):
        de = self.ud.pop('_drawelement', None)
        if de is not None:
            win.canvas.after.remove(de[0])
            win.canvas.after.remove(de[1])