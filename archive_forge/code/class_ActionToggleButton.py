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
class ActionToggleButton(ActionItem, ToggleButton):
    """
    ActionToggleButton class, see module documentation for more information.
    """
    icon = StringProperty(None, allownone=True)
    '\n    Source image to use when the Button is part of the ActionBar. If the\n    Button is in a group, the text will be preferred.\n    '