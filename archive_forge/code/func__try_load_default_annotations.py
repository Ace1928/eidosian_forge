from json import load
from os.path import exists
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
from kivy.animation import Animation
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.video import Image
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.clock import Clock
def _try_load_default_annotations(self, *largs):
    if not self.annotations:
        filename = self.source.rsplit('.', 1)
        annotations = filename[0] + '.jsa'
        if exists(annotations):
            self._load_annotations(annotations)