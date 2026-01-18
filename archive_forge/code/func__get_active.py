from kivy.properties import AliasProperty, StringProperty, ColorProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.widget import Widget
def _get_active(self):
    return self.state == 'down'