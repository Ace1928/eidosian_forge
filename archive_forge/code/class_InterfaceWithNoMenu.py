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
class InterfaceWithNoMenu(ContentPanel):
    """The interface widget used by :class:`SettingsWithNoMenu`. It
    stores and displays a single settings panel.

    This widget is considered internal and is not documented. See the
    :class:`ContentPanel` for information on defining your own content
    widget.

    """

    def add_widget(self, *args, **kwargs):
        if self.container is not None and len(self.container.children) > 0:
            raise Exception('ContentNoMenu cannot accept more than one settings panel')
        super(InterfaceWithNoMenu, self).add_widget(*args, **kwargs)