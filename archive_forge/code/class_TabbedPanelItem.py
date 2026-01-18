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
class TabbedPanelItem(TabbedPanelHeader):
    """This is a convenience class that provides a header of type
    TabbedPanelHeader and links it with the content automatically. Thus
    facilitating you to simply do the following in kv language:

    .. code-block:: kv

        <TabbedPanel>:
            # ...other settings
            TabbedPanelItem:
                BoxLayout:
                    Label:
                        text: 'Second tab content area'
                    Button:
                        text: 'Button that does nothing'

    .. versionadded:: 1.5.0
    """

    def add_widget(self, widget, *args, **kwargs):
        self.content = widget
        if not self.parent:
            return
        panel = self.parent.tabbed_panel
        if panel.current_tab == self:
            panel.switch_to(self)

    def remove_widget(self, *args, **kwargs):
        self.content = None
        if not self.parent:
            return
        panel = self.parent.tabbed_panel
        if panel.current_tab == self:
            panel.remove_widget(*args, **kwargs)