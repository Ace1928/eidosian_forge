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
class SettingPath(SettingItem):
    """Implementation of a Path setting on top of a :class:`SettingItem`.
    It is visualized with a :class:`~kivy.uix.label.Label` widget that, when
    clicked, will open a :class:`~kivy.uix.popup.Popup` with a
    :class:`~kivy.uix.filechooser.FileChooserListView` so the user can enter
    a custom value.

    .. versionadded:: 1.1.0
    """
    popup = ObjectProperty(None, allownone=True)
    '(internal) Used to store the current popup when it is shown.\n\n    :attr:`popup` is an :class:`~kivy.properties.ObjectProperty` and defaults\n    to None.\n    '
    textinput = ObjectProperty(None)
    '(internal) Used to store the current textinput from the popup and\n    to listen for changes.\n\n    :attr:`textinput` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    show_hidden = BooleanProperty(False)
    "Whether to show 'hidden' filenames. What that means is\n    operating-system-dependent.\n\n    :attr:`show_hidden` is an :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n\n    .. versionadded:: 1.10.0\n    "
    dirselect = BooleanProperty(True)
    'Whether to allow selection of directories.\n\n    :attr:`dirselect` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n\n    .. versionadded:: 1.10.0\n    '

    def on_panel(self, instance, value):
        if value is None:
            return
        self.fbind('on_release', self._create_popup)

    def _dismiss(self, *largs):
        if self.textinput:
            self.textinput.focus = False
        if self.popup:
            self.popup.dismiss()
        self.popup = None

    def _validate(self, instance):
        self._dismiss()
        value = self.textinput.selection
        if not value:
            return
        self.value = os.path.realpath(value[0])

    def _create_popup(self, instance):
        content = BoxLayout(orientation='vertical', spacing=5)
        popup_width = min(0.95 * Window.width, dp(500))
        self.popup = popup = Popup(title=self.title, content=content, size_hint=(None, 0.9), width=popup_width)
        initial_path = self.value or os.getcwd()
        self.textinput = textinput = FileChooserListView(path=initial_path, size_hint=(1, 1), dirselect=self.dirselect, show_hidden=self.show_hidden)
        textinput.bind(on_path=self._validate)
        content.add_widget(textinput)
        content.add_widget(SettingSpacer())
        btnlayout = BoxLayout(size_hint_y=None, height='50dp', spacing='5dp')
        btn = Button(text='Ok')
        btn.bind(on_release=self._validate)
        btnlayout.add_widget(btn)
        btn = Button(text='Cancel')
        btn.bind(on_release=self._dismiss)
        btnlayout.add_widget(btn)
        content.add_widget(btnlayout)
        popup.open()