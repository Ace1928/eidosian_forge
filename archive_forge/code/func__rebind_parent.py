from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
def _rebind_parent(self, instance, new_parent):
    if self._bound_parent is not None:
        self._bound_parent.unbind(size=self.rescale_parent_proportion)
    if self.parent is not None:
        new_parent.bind(size=self.rescale_parent_proportion)
    self._bound_parent = new_parent
    self.rescale_parent_proportion()