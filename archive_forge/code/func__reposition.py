from kivy.uix.scrollview import ScrollView
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.config import Config
def _reposition(self, *largs):
    win = self._win
    if not win:
        return
    widget = self.attach_to
    if not widget or not widget.get_parent_window():
        return
    wx, wy = widget.to_window(*widget.pos)
    wright, wtop = widget.to_window(widget.right, widget.top)
    if self.auto_width:
        self.width = wright - wx
    x = wx
    if x + self.width > win.width:
        x = win.width - self.width
    if x < 0:
        x = 0
    self.x = x
    if self.max_height is not None:
        height = min(self.max_height, self.container.minimum_height)
    else:
        height = self.container.minimum_height
    h_bottom = wy - height
    h_top = win.height - (wtop + height)
    if h_bottom > 0:
        self.top = wy
        self.height = height
    elif h_top > 0:
        self.y = wtop
        self.height = height
    elif h_top < h_bottom:
        self.top = self.height = wy
    else:
        self.y = wtop
        self.height = win.height - wtop