from kivy.clock import Clock
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.config import Config
from kivy.metrics import sp
from functools import partial
def _do_touch_up(self, touch, *largs):
    super(DragBehavior, self).on_touch_up(touch)
    for x in touch.grab_list[:]:
        touch.grab_list.remove(x)
        x = x()
        if not x:
            continue
        touch.grab_current = x
        super(DragBehavior, self).on_touch_up(touch)
    touch.grab_current = None