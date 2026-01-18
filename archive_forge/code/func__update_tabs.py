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
def _update_tabs(self, *l):
    self_content = self.content
    if not self_content:
        return
    tab_pos = self.tab_pos
    tab_layout = self._tab_layout
    tab_layout.clear_widgets()
    scrl_v = ScrollView(size_hint=(None, 1), always_overscroll=False, bar_width=self.bar_width, scroll_type=self.scroll_type)
    tabs = self._tab_strip
    parent = tabs.parent
    if parent:
        parent.remove_widget(tabs)
    scrl_v.add_widget(tabs)
    scrl_v.pos = (0, 0)
    self_update_scrollview = self._update_scrollview
    if self._partial_update_scrollview is not None:
        tabs.unbind(width=self._partial_update_scrollview)
    self._partial_update_scrollview = partial(self_update_scrollview, scrl_v)
    tabs.bind(width=self._partial_update_scrollview)
    super(TabbedPanel, self).clear_widgets()
    tab_height = self.tab_height
    widget_list = []
    tab_list = []
    pos_letter = tab_pos[0]
    if pos_letter == 'b' or pos_letter == 't':
        self.cols = 1
        self.rows = 2
        tab_layout.rows = 1
        tab_layout.cols = 3
        tab_layout.size_hint = (1, None)
        tab_layout.height = tab_height + tab_layout.padding[1] + tab_layout.padding[3] + dp(2)
        self_update_scrollview(scrl_v)
        if pos_letter == 'b':
            if tab_pos == 'bottom_mid':
                tab_list = (Widget(), scrl_v, Widget())
                widget_list = (self_content, tab_layout)
            else:
                if tab_pos == 'bottom_left':
                    tab_list = (scrl_v, Widget(), Widget())
                elif tab_pos == 'bottom_right':
                    tab_list = (Widget(), Widget(), scrl_v)
                widget_list = (self_content, tab_layout)
        else:
            if tab_pos == 'top_mid':
                tab_list = (Widget(), scrl_v, Widget())
            elif tab_pos == 'top_left':
                tab_list = (scrl_v, Widget(), Widget())
            elif tab_pos == 'top_right':
                tab_list = (Widget(), Widget(), scrl_v)
            widget_list = (tab_layout, self_content)
    elif pos_letter == 'l' or pos_letter == 'r':
        self.cols = 2
        self.rows = 1
        tab_layout.rows = 3
        tab_layout.cols = 1
        tab_layout.size_hint = (None, 1)
        tab_layout.width = tab_height
        scrl_v.height = tab_height
        self_update_scrollview(scrl_v)
        rotation = 90 if tab_pos[0] == 'l' else -90
        sctr = Scatter(do_translation=False, rotation=rotation, do_rotation=False, do_scale=False, size_hint=(None, None), auto_bring_to_front=False, size=scrl_v.size)
        sctr.add_widget(scrl_v)
        lentab_pos = len(tab_pos)
        if tab_pos[lentab_pos - 4:] == '_top':
            sctr.bind(pos=partial(self._update_top, sctr, 'top', None))
            tab_list = (sctr,)
        elif tab_pos[lentab_pos - 4:] == '_mid':
            sctr.bind(pos=partial(self._update_top, sctr, 'mid', scrl_v.width))
            tab_list = (Widget(), sctr, Widget())
        elif tab_pos[lentab_pos - 7:] == '_bottom':
            tab_list = (Widget(), Widget(), sctr)
        if pos_letter == 'l':
            widget_list = (tab_layout, self_content)
        else:
            widget_list = (self_content, tab_layout)
    add = tab_layout.add_widget
    for widg in tab_list:
        add(widg)
    add = self.add_widget
    for widg in widget_list:
        add(widg)