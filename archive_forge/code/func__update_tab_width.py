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
def _update_tab_width(self, *l):
    if self.tab_width:
        for tab in self.tab_list:
            tab.size_hint_x = 1
        tsw = self.tab_width * len(self._tab_strip.children)
    else:
        tsw = 0
        for tab in self.tab_list:
            if tab.size_hint_x:
                tab.size_hint_x = 1
                tsw += 100
            else:
                tsw += tab.width
    self._tab_strip.width = tsw
    self._reposition_tabs()