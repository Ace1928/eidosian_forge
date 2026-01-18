from kivy.animation import Animation
from kivy.properties import (
from kivy.uix.anchorlayout import AnchorLayout
def _handle_keyboard(self, _window, key, *_args):
    if key == 27 and self.auto_dismiss:
        self.dismiss()
        return True