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
def _update_scrollview(self, scrl_v, *l):
    self_tab_pos = self.tab_pos
    self_tabs = self._tab_strip
    if self_tab_pos[0] == 'b' or self_tab_pos[0] == 't':
        scrl_v.width = min(self.width, self_tabs.width)
        scrl_v.top += 1
        scrl_v.top -= 1
    else:
        scrl_v.width = min(self.height, self_tabs.width)
        self_tabs.pos = (0, 0)