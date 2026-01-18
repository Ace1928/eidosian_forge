from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.config import Config
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty, \
from kivy.metrics import sp
from kivy.lang import Builder
from functools import partial
def _layout_group(self):
    super_add = super(ActionView, self).add_widget
    self._state = 'group'
    self._clear_all()
    if not self.action_previous.parent:
        super_add(self.action_previous)
    if len(self._list_action_items) > 1:
        for child in self._list_action_items[1:]:
            super_add(child)
            child.inside_group = False
    for group in self._list_action_group:
        super_add(group)
        group.show_group()
    self.overflow_group.show_default_items(self)