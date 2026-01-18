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
class VideoPlayerPlayPause(Image):
    video = ObjectProperty(None)

    def on_touch_down(self, touch):
        """.. versionchanged:: 1.4.0"""
        if self.collide_point(*touch.pos):
            if self.video.state == 'play':
                self.video.state = 'pause'
            else:
                self.video.state = 'play'
            return True