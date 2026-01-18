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
def _toggle_dropdown(self, *largs):
    ddn = self._dropdown
    ddn.size_hint_x = None
    if not ddn.container:
        return
    children = ddn.container.children
    if children:
        ddn.width = self.dropdown_width or max(self.width, max((c.pack_width for c in children)))
    else:
        ddn.width = self.width
    for item in children:
        item.size_hint_y = None
        item.height = max([self.height, sp(48)])
        item.bind(on_release=ddn.dismiss)