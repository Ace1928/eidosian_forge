import json
import os
import kivy.utils as utils
from kivy.factory import Factory
from kivy.metrics import dp
from kivy.config import ConfigParser
from kivy.animation import Animation
from kivy.compat import string_types, text_type
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanelHeader
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty, ListProperty, \
class SettingBoolean(SettingItem):
    """Implementation of a boolean setting on top of a :class:`SettingItem`.
    It is visualized with a :class:`~kivy.uix.switch.Switch` widget.
    By default, 0 and 1 are used for values: you can change them by setting
    :attr:`values`.
    """
    values = ListProperty(['0', '1'])
    'Values used to represent the state of the setting. If you want to use\n    "yes" and "no" in your ConfigParser instance::\n\n        SettingBoolean(..., values=[\'no\', \'yes\'])\n\n    .. warning::\n\n        You need a minimum of two values, the index 0 will be used as False,\n        and index 1 as True\n\n    :attr:`values` is a :class:`~kivy.properties.ListProperty` and defaults to\n    [\'0\', \'1\']\n    '