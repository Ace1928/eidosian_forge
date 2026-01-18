from functools import partial
from kivy.clock import Clock
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty, OptionProperty, \
def _switch_to_first_tab(self, *l):
    ltl = len(self.tab_list) - 1
    if ltl > -1:
        self._current_tab = dt = self._original_tab = self.tab_list[ltl]
        self.switch_to(dt)