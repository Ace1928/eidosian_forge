from kivy.properties import AliasProperty, StringProperty, ColorProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.widget import Widget
def _set_active(self, value):
    self.state = 'down' if value else 'normal'