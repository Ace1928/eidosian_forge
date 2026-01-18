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
class SettingOptions(SettingItem):
    """Implementation of an option list on top of a :class:`SettingItem`.
    It is visualized with a :class:`~kivy.uix.label.Label` widget that, when
    clicked, will open a :class:`~kivy.uix.popup.Popup` with a
    list of options from which the user can select.
    """
    options = ListProperty([])
    'List of all availables options. This must be a list of "string" items.\n    Otherwise, it will crash. :)\n\n    :attr:`options` is a :class:`~kivy.properties.ListProperty` and defaults\n    to [].\n    '
    popup = ObjectProperty(None, allownone=True)
    '(internal) Used to store the current popup when it is shown.\n\n    :attr:`popup` is an :class:`~kivy.properties.ObjectProperty` and defaults\n    to None.\n    '

    def on_panel(self, instance, value):
        if value is None:
            return
        self.fbind('on_release', self._create_popup)

    def _set_option(self, instance):
        self.value = instance.text
        self.popup.dismiss()

    def _create_popup(self, instance):
        content = BoxLayout(orientation='vertical', spacing='5dp')
        popup_width = min(0.95 * Window.width, dp(500))
        self.popup = popup = Popup(content=content, title=self.title, size_hint=(None, None), size=(popup_width, '400dp'))
        popup.height = len(self.options) * dp(55) + dp(150)
        content.add_widget(Widget(size_hint_y=None, height=1))
        uid = str(self.uid)
        for option in self.options:
            state = 'down' if option == self.value else 'normal'
            btn = ToggleButton(text=option, state=state, group=uid)
            btn.bind(on_release=self._set_option)
            content.add_widget(btn)
        content.add_widget(SettingSpacer())
        btn = Button(text='Cancel', size_hint_y=None, height=dp(50))
        btn.bind(on_release=popup.dismiss)
        content.add_widget(btn)
        popup.open()