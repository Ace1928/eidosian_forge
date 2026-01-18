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
class InterfaceWithSpinner(BoxLayout):
    """A settings interface that displays a spinner at the top for
    switching between panels.

    The workings of this class are considered internal and are not
    documented. See :meth:`InterfaceWithSidebar` for
    information on implementing your own interface class.

    """
    __events__ = ('on_close',)
    menu = ObjectProperty()
    '(internal) A reference to the sidebar menu widget.\n\n    :attr:`menu` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    content = ObjectProperty()
    '(internal) A reference to the panel display widget (a\n    :class:`ContentPanel`).\n\n    :attr:`menu` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n\n    '

    def __init__(self, *args, **kwargs):
        super(InterfaceWithSpinner, self).__init__(*args, **kwargs)
        self.menu.close_button.bind(on_release=lambda j: self.dispatch('on_close'))

    def add_panel(self, panel, name, uid):
        """This method is used by Settings to add new panels for possible
        display. Any replacement for ContentPanel *must* implement
        this method.

        :Parameters:
            `panel`: :class:`SettingsPanel`
                It should be stored and the interface should provide a way to
                switch between panels.
            `name`:
                The name of the panel as a string. It may be used to represent
                the panel but may not be unique.
            `uid`:
                A unique int identifying the panel. It should be used to
                identify and switch between panels.

        """
        self.content.add_panel(panel, name, uid)
        self.menu.add_item(name, uid)

    def on_close(self, *args):
        pass