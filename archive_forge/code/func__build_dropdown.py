from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import ListProperty, ObjectProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
def _build_dropdown(self, *largs):
    if self._dropdown:
        self._dropdown.unbind(on_select=self._on_dropdown_select)
        self._dropdown.unbind(on_dismiss=self._close_dropdown)
        self._dropdown.dismiss()
        self._dropdown = None
    cls = self.dropdown_cls
    if isinstance(cls, string_types):
        cls = Factory.get(cls)
    self._dropdown = cls()
    self._dropdown.bind(on_select=self._on_dropdown_select)
    self._dropdown.bind(on_dismiss=self._close_dropdown)
    self._update_dropdown()