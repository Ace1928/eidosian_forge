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
class TabbedPanelStrip(GridLayout):
    """A strip intended to be used as background for Heading/Tab.
    This does not cover the blank areas in case the tabs don't cover
    the entire width/height of the TabbedPanel(use :class:`StripLayout`
    for that).
    """
    tabbed_panel = ObjectProperty(None)
    'Link to the panel that the tab strip is a part of.\n\n    :attr:`tabbed_panel` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None .\n    '