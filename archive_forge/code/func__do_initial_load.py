from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def _do_initial_load(self, *largs):
    if not self.load_func:
        return
    self._do_node_load(None)