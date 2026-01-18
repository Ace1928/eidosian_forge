from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
def _do_size(self, instance, value):
    if self.sizable_from[0] in ('l', 'r'):
        self.width = max(self.min_size, min(self.width, self.max_size))
    else:
        self.height = max(self.min_size, min(self.height, self.max_size))