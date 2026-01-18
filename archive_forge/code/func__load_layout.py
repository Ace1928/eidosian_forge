from kivy import kivy_data_dir
from kivy.vector import Vector
from kivy.config import Config
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
from kivy.logger import Logger
from kivy.graphics import Color, BorderImage, Canvas
from kivy.core.image import Image
from kivy.resources import resource_find
from kivy.clock import Clock
from io import open
from os.path import join, splitext, basename
from os import listdir
from json import loads
def _load_layout(self, *largs):
    if self._trigger_load_layouts.is_triggered:
        self._load_layouts()
        self._trigger_load_layouts.cancel()
    value = self.layout
    available_layouts = self.available_layouts
    if self.layout[-5:] == '.json':
        if value not in available_layouts:
            fn = resource_find(self.layout)
            self._load_layout_fn(fn, self.layout)
    if not available_layouts:
        return
    if value not in available_layouts and value != 'qwerty':
        Logger.error('Vkeyboard: <%s> keyboard layout mentioned in conf file was not found, fallback on qwerty' % value)
        self.layout = 'qwerty'
    self.refresh(True)