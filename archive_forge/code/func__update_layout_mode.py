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
def _update_layout_mode(self, *l):
    mode = self.have_capslock != self.have_shift
    mode = 'shift' if mode else 'normal'
    if self.have_special:
        mode = 'special'
    if mode != self.layout_mode:
        self.layout_mode = mode
        self.refresh(False)