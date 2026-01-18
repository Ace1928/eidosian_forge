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
def _update_bubble(self, *l):
    seek = self.seek
    if self.seek is None:
        if self.video.duration == 0:
            seek = 0
        else:
            seek = self.video.position / self.video.duration
    d = self.video.duration * seek
    minutes = int(d / 60)
    seconds = int(d - minutes * 60)
    self.bubble_label.text = '%d:%02d' % (minutes, seconds)
    self.bubble.center_x = self.x + seek * self.width
    self.bubble.y = self.top