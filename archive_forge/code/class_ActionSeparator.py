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
class ActionSeparator(ActionItem, Widget):
    """
    ActionSeparator class, see module documentation for more information.
    """
    background_image = StringProperty('atlas://data/images/defaulttheme/separator')
    "\n    Background image for the separators default graphical representation.\n\n    :attr:`background_image` is a :class:`~kivy.properties.StringProperty`\n    and defaults to 'atlas://data/images/defaulttheme/separator'.\n    "