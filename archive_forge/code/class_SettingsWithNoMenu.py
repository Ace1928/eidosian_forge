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
class SettingsWithNoMenu(Settings):
    """A settings widget that displays a single settings panel with *no*
    Close button. It will not accept more than one Settings panel. It
    is intended for use in programs with few enough settings that a
    full panel switcher is not useful.

    .. warning::

        This Settings panel does *not* provide a Close
        button, and so it is impossible to leave the settings screen
        unless you also add other behavior or override
        :meth:`~kivy.app.App.display_settings` and
        :meth:`~kivy.app.App.close_settings`.

    """

    def __init__(self, *args, **kwargs):
        self.interface_cls = InterfaceWithNoMenu
        super(SettingsWithNoMenu, self).__init__(*args, **kwargs)