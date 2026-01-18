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
class SettingColor(SettingItem):
    """Implementation of a color setting on top of a :class:`SettingItem`.
    It is visualized with a :class:`~kivy.uix.label.Label` widget and a
    colored canvas rectangle that, when clicked, will open a
    :class:`~kivy.uix.popup.Popup` with a
    :class:`~kivy.uix.colorpicker.ColorPicker` so the user can choose a color.

    .. versionadded:: 2.0.1
    """
    popup = ObjectProperty(None, allownone=True)
    "(internal) Used to store the current popup when it's shown.\n\n    :attr:`popup` is an :class:`~kivy.properties.ObjectProperty` and defaults\n    to None.\n    "

    def on_panel(self, instance, value):
        if value is None:
            return
        self.bind(on_release=self._create_popup)

    def _dismiss(self, *largs):
        if self.popup:
            self.popup.dismiss()
        self.popup = None

    def _validate(self, instance):
        self._dismiss()
        value = utils.get_hex_from_color(self.colorpicker.color)
        self.value = value

    def _create_popup(self, instance):
        content = BoxLayout(orientation='vertical', spacing='5dp')
        popup_width = min(0.95 * Window.width, dp(500))
        self.popup = popup = Popup(title=self.title, content=content, size_hint=(None, 0.9), width=popup_width)
        self.colorpicker = colorpicker = ColorPicker(color=utils.get_color_from_hex(self.value))
        colorpicker.bind(on_color=self._validate)
        self.colorpicker = colorpicker
        content.add_widget(colorpicker)
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