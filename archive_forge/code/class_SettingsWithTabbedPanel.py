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
class SettingsWithTabbedPanel(Settings):
    """A settings widget that displays settings panels as pages in a
    :class:`~kivy.uix.tabbedpanel.TabbedPanel`.
    """
    __events__ = ('on_close',)

    def __init__(self, *args, **kwargs):
        self.interface_cls = InterfaceWithTabbedPanel
        super(SettingsWithTabbedPanel, self).__init__(*args, **kwargs)

    def on_close(self, *args):
        pass