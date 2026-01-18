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
class TabbedPanelHeader(ToggleButton):
    """A Base for implementing a Tabbed Panel Head. A button intended to be
    used as a Heading/Tab for a TabbedPanel widget.

    You can use this TabbedPanelHeader widget to add a new tab to a
    TabbedPanel.
    """
    content = ObjectProperty(None, allownone=True)
    'Content to be loaded when this tab header is selected.\n\n    :attr:`content` is an :class:`~kivy.properties.ObjectProperty` and defaults\n    to None.\n    '

    def on_touch_down(self, touch):
        if self.state == 'down':
            for child in self.children:
                child.dispatch('on_touch_down', touch)
            return
        else:
            super(TabbedPanelHeader, self).on_touch_down(touch)

    def on_release(self, *largs):
        if self.parent:
            self.parent.tabbed_panel.switch_to(self)
        else:
            self.panel.switch_to(self.panel.current_tab)