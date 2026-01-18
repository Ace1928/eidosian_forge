from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import ListProperty, ObjectProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
def _update_dropdown(self, *largs):
    dp = self._dropdown
    cls = self.option_cls
    values = self.values
    text_autoupdate = self.text_autoupdate
    if isinstance(cls, string_types):
        cls = Factory.get(cls)
    dp.clear_widgets()
    for value in values:
        item = cls(text=value)
        item.height = self.height if self.sync_height else item.height
        item.bind(on_release=lambda option: dp.select(option.text))
        dp.add_widget(item)
    if text_autoupdate:
        if values:
            if not self.text or self.text not in values:
                self.text = values[0]
        else:
            self.text = ''