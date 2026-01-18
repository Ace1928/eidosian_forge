from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _on_focus(self, instance, value, *largs):
    if self.keyboard_mode == 'auto':
        if value:
            self._bind_keyboard()
        else:
            self._unbind_keyboard()