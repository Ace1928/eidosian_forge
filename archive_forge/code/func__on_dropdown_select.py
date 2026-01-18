from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import ListProperty, ObjectProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
def _on_dropdown_select(self, instance, data, *largs):
    self.text = data
    self.is_open = False