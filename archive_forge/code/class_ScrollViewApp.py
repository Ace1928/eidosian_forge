from functools import partial
from kivy.animation import Animation
from kivy.compat import string_types
from kivy.config import Config
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.stencilview import StencilView
from kivy.metrics import dp
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.properties import NumericProperty, BooleanProperty, AliasProperty, \
from kivy.uix.behaviors import FocusBehavior
class ScrollViewApp(App):

    def build(self):
        layout1 = GridLayout(cols=4, spacing=10, size_hint=(None, None))
        layout1.bind(minimum_height=layout1.setter('height'), minimum_width=layout1.setter('width'))
        for i in range(40):
            btn = Button(text=str(i), size_hint=(None, None), size=(200, 100))
            layout1.add_widget(btn)
        scrollview1 = ScrollView(bar_width='2dp', smooth_scroll_end=10)
        scrollview1.add_widget(layout1)
        layout2 = GridLayout(cols=4, spacing=10, size_hint=(None, None))
        layout2.bind(minimum_height=layout2.setter('height'), minimum_width=layout2.setter('width'))
        for i in range(40):
            btn = Button(text=str(i), size_hint=(None, None), size=(200, 100))
            layout2.add_widget(btn)
        scrollview2 = ScrollView(scroll_type=['bars'], bar_width='9dp', scroll_wheel_distance=100)
        scrollview2.add_widget(layout2)
        root = GridLayout(cols=2)
        root.add_widget(scrollview1)
        root.add_widget(scrollview2)
        return root