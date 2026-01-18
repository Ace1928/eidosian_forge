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
class ContentPanel(ScrollView):
    """A class for displaying settings panels. It displays a single
    settings panel at a time, taking up the full size and shape of the
    ContentPanel. It is used by :class:`InterfaceWithSidebar` and
    :class:`InterfaceWithSpinner` to display settings.

    """
    panels = DictProperty({})
    '(internal) Stores a dictionary mapping settings panels to their uids.\n\n    :attr:`panels` is a :class:`~kivy.properties.DictProperty` and\n    defaults to {}.\n\n    '
    container = ObjectProperty()
    '(internal) A reference to the GridLayout that contains the\n    settings panel.\n\n    :attr:`container` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n\n    '
    current_panel = ObjectProperty(None)
    '(internal) A reference to the current settings panel.\n\n    :attr:`current_panel` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n\n    '
    current_uid = NumericProperty(0)
    '(internal) A reference to the uid of the current settings panel.\n\n    :attr:`current_uid` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to 0.\n\n    '

    def add_panel(self, panel, name, uid):
        """This method is used by Settings to add new panels for possible
        display. Any replacement for ContentPanel *must* implement
        this method.

        :Parameters:
            `panel`: :class:`SettingsPanel`
                It should be stored and displayed when requested.
            `name`:
                The name of the panel as a string. It may be used to represent
                the panel.
            `uid`:
                A unique int identifying the panel. It should be stored and
                used to identify panels when switching.

        """
        self.panels[uid] = panel
        if not self.current_uid:
            self.current_uid = uid

    def on_current_uid(self, *args):
        """The uid of the currently displayed panel. Changing this will
        automatically change the displayed panel.

        :Parameters:
            `uid`:
                A panel uid. It should be used to retrieve and display
                a settings panel that has previously been added with
                :meth:`add_panel`.

        """
        uid = self.current_uid
        if uid in self.panels:
            if self.current_panel is not None:
                self.remove_widget(self.current_panel)
            new_panel = self.panels[uid]
            self.add_widget(new_panel)
            self.current_panel = new_panel
            return True
        return False

    def add_widget(self, *args, **kwargs):
        if self.container is None:
            super(ContentPanel, self).add_widget(*args, **kwargs)
        else:
            self.container.add_widget(*args, **kwargs)

    def remove_widget(self, *args, **kwargs):
        self.container.remove_widget(*args, **kwargs)