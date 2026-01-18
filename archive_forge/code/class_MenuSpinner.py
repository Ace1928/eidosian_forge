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
class MenuSpinner(BoxLayout):
    """The menu class used by :class:`SettingsWithSpinner`. It provides a
    sidebar with an entry for each settings panel.

    This widget is considered internal and is not documented. See
    :class:`MenuSidebar` for information on menus and creating your own menu
    class.

    """
    selected_uid = NumericProperty(0)
    close_button = ObjectProperty(0)
    spinner = ObjectProperty()
    panel_names = DictProperty({})
    spinner_text = StringProperty()
    close_button = ObjectProperty()

    def add_item(self, name, uid):
        values = self.spinner.values
        if name in values:
            i = 2
            while name + ' {}'.format(i) in values:
                i += 1
            name = name + ' {}'.format(i)
        self.panel_names[name] = uid
        self.spinner.values.append(name)
        if not self.spinner.text:
            self.spinner.text = name

    def on_spinner_text(self, *args):
        text = self.spinner_text
        self.selected_uid = self.panel_names[text]