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
def _load_layout_fn(self, fn, name):
    available_layouts = self.available_layouts
    if fn[-5:] != '.json':
        return
    with open(fn, 'r', encoding='utf-8') as fd:
        json_content = fd.read()
        layout = loads(json_content)
    available_layouts[name] = layout