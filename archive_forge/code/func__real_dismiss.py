from kivy.uix.scrollview import ScrollView
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.config import Config
def _real_dismiss(self, *largs):
    if self.parent:
        self.parent.remove_widget(self)
    if self.attach_to:
        self.attach_to.unbind(pos=self._reposition, size=self._reposition)
        self.attach_to = None
    self.dispatch('on_dismiss')