from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _ensure_keyboard(self):
    if self._keyboard is None:
        self._requested_keyboard = True
        keyboard = self._keyboard = EventLoop.window.request_keyboard(self._keyboard_released, self, input_type=self.input_type, keyboard_suggestions=self.keyboard_suggestions)
        keyboards = FocusBehavior._keyboards
        if keyboard not in keyboards:
            keyboards[keyboard] = None