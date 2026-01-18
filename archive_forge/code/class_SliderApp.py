from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, AliasProperty, OptionProperty,
class SliderApp(App):

    def build(self):
        return Slider(padding=25)