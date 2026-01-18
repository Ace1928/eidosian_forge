from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (ObjectProperty, StringProperty,
from kivy.uix.widget import Widget
from kivy.logger import Logger
Minimum space to use for the title of each item. This value is
    automatically set for each child every time the layout event occurs.

    :attr:`min_space` is a :class:`~kivy.properties.NumericProperty` and
    defaults to 44 (px).
    